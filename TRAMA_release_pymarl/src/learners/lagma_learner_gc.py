import copy
from re import I
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
from modules.mixers.dmaq_qatten import DMAQ_QattenMixer
from modules.vae import REGISTRY as vae_REGISTRY
import torch as th
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import RMSprop, Adam
from utils.torch_utils import to_cuda
import time 
import numpy as np

class LAGMAGCLearner:
    def __init__(self, mac, vqvae, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        self.device = self.args.device
        self.bs     = args.batch_size
        
        self.default_node = self.args.n_codes
        
        self.nk = int(self.args.n_codes / self.args.n_cluster)

        self.state_dim = scheme["state"]["vshape"]
        #input_shape    = self._get_input_shape(scheme) # input_shape includes agent-id (if available)
        
        self.params = list(mac.parameters())
        if self.args.use_vqvae:
            #self.vae = vae_REGISTRY[self.args.vae](self.state_dim, self.state_dim, self.args)
            self.vae = vqvae
            self.vae_params = list(self.vae.parameters())
            self.vae_optimizer = Adam(params=self.vae_params, lr=args.lr)
        
        self.update_vqvae = False
        self.update_codebook = False
        self.last_vqvae_update_episode = 0
        self.last_codebook_update_episode = 0
        self.last_target_update_episode = 0
        
        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            elif args.mixer == 'dmaq_qatten':
                self.mixer = DMAQ_QattenMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        
        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.vae_log_t = self.log_stats_t = -self.args.learner_log_interval - 1
        
        self.vae_losses      = th.tensor(0.0).to(self.args.device)   
        self.ce_losses       = th.tensor(0.0).to(self.args.device)   
        self.vq_losses       = th.tensor(0.0).to(self.args.device)   
        self.commit_losses   = th.tensor(0.0).to(self.args.device)   
        self.coverage_losses = th.tensor(0.0).to(self.args.device)   
        
        self.criterion = nn.CrossEntropyLoss()        

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, seq_centroid=None, RBS_cluster=None, buffer_seq_labels=None, f_classifier=None):
        # Get the relevant quantities        
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        #selected_goals   = batch["goals"][:, :-1] # [bs,t,n_agent,1] # prediction made by agents
        selected_goals   = batch["goals"] # [bs,t,n_agent,1] # prediction made by agents        
        terminated       = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        actions_onehot = batch["actions_onehot"][:, :-1]
        win_flag = batch["flag_win"].squeeze(-1)

        min_timestep_true = th.min(th.nonzero( terminated.squeeze(-1) == 1)[:,1] ) # [bs] --> 1
        if min_timestep_true <= 10:
            checkpoint = 1
        mask_pred_accuracy = mask.expand_as(th.zeros([mask.size()[0],mask.size()[1],self.args.n_agents])).permute(0,2,1).reshape(-1,mask.size()[1]) # [bs,t,n_agents] --> [bs*n_agent,t]
        
        flag_des_trj = th.any( win_flag != 0, dim=1 ) # [bs]
       
        if buffer_seq_labels is not None:
            seq_labels = buffer_seq_labels
        else:
            seq_labels = None

        mean_acc_t0 = None

        if th.any(flag_des_trj != 0 ):
            find_desirable_trj = 1

        # backward reward and reward sum generation for codebook update                
        rewards_th = th.tensor(batch["reward"]).to(self.device).squeeze(-1)
        sum_rewards = th.sum(th.tensor( rewards_th ).to(self.device), axis=1) # [bs]
        reward_tgo  = th.zeros_like(rewards_th ).to(self.device)

        #.. reverse sequence for reward-to-go computation
        for t in range(batch.max_seq_length-1, -1, -1):
            if t == batch.max_seq_length-1:
                reward_tgo[:, t] = rewards_th[:,t]
            else:
                reward_tgo[:, t] = rewards_th[:,t] + self.args.gamma*reward_tgo[:, t+1]
        
        if self.args.vqvae_update_type == 1: # training with the current batch from replay buffer
            if ((episode_num - self.last_vqvae_update_episode) / self.args.vqvae_update_interval >= 1.0):
                self.update_vqvae = True
                self.last_vqvae_update_episode = episode_num

                if (t_env >= self.args.vqvae_training_stop):
                    self.update_vqvae = False

            else:
                self.update_vqvae = False       
                if t_env <= self.args.buffer_update_time: # update vqvae at early trainig time
                    self.update_vqvae = True

            if ((episode_num - self.last_codebook_update_episode) / self.args.codebook_update_interval >= 1.0):
                self.update_codebook = True
                self.last_codebook_update_episode = episode_num
            else:
                self.update_codebook = False       
                if t_env <= self.args.buffer_update_time: # update codebook at early trainig time
                    self.update_codebook = True

        else:
            self.update_vqvae = False       

            if ((episode_num - self.last_codebook_update_episode) / self.args.codebook_update_interval >= 1.0):
                self.update_codebook = True
                self.last_codebook_update_episode = episode_num
            else:
                self.update_codebook = False       
                if t_env <= self.args.buffer_update_time: # update codebook at early trainig time
                    self.update_codebook = True
                
        # Calculate estimated Q-Values
        mac_out = []
        goal_prob_logits_out = []

        self.mac.init_hidden(batch.batch_size)

        #vae_losses      = th.tensor(0.0).to(self.args.device)        
        #.. for monitoring
        ce_losses       = th.tensor(0.0).to(self.args.device)
        vq_losses       = th.tensor(0.0).to(self.args.device)
        commit_losses   = th.tensor(0.0).to(self.args.device)
        coverage_losses = th.tensor(0.0).to(self.args.device)
        
        visit_nodes =[]
        if self.args.flag_batch_wise_vqvae_loss:
            vae_losses  = [] # need masking
        else:
            vae_losses  = th.tensor(0.0).to(self.args.device)

        buf_state_input = []
        buf_recon       = []
        buf_z_e         = []
        buf_latent_emb  = []
        buf_Cqt_hat     = []
        # buf_Cqt_target  = []

        #ts = time.time()
        # seq sampling loop ==============================================================================
        for t in range(batch.max_seq_length):
            if self.args.mac == "lagma_gc_mac":                
                agent_outs, goal_prob_logits, _ = self.mac.forward(batch, t=t, goal_latent=None, selected_goal=selected_goals[:,t].squeeze(-1).reshape(-1))
                mac_out.append(agent_outs)
                goal_prob_logits_out.append(goal_prob_logits)
            
            else:
                agent_outs = self.mac.forward(batch, t=t)
                mac_out.append(agent_outs)
                
            #.. add vq-vae estimation -----
            if self.args.use_vqvae:
                state_input = th.tensor(batch["state"][:, t]).to(self.device)
            
                if self.args.recon_type==3:
                    timestep = th.tensor( [float(t) / float(self.args.max_seq_length)] ).repeat(self.args.batch_size).unsqueeze(-1).to(self.args.device)
                    embed_input = th.cat( [state_input, timestep], dim=1) # [bs,dim]
                    recon, z_e, latent_emb, argmin, Cqt_hat = self.vae(embed_input, timestep = timestep)                    
                elif self.args.recon_type==2:
                    recon, z_e, latent_emb, argmin, Cqt_hat = self.vae(state_input)                    
                else:
                    recon, z_e, latent_emb, argmin = self.vae(state_input) # [bs,dim]
                     
                # manage zero state vector
                if self.args.flag_zero_state_management==True:
                    sums = th.sum(state_input,dim=1)
                    #zero_index = th.nonzero(sums==0, as_tuple=False).squeeze()
                    zero_index = th.nonzero(sums==0, as_tuple=False)
                    if len(zero_index) > 0 :
                        argmin[zero_index] = self.args.n_codes
                        
                visit_nodes.append(argmin)
                buf_state_input.append(state_input)
                buf_recon.append(recon)      
                buf_z_e.append(z_e)        
                buf_latent_emb.append(latent_emb) 
                if self.args.recon_type == 2 or self.args.recon_type == 3:
                    buf_Cqt_hat.append(Cqt_hat)
                    # buf_Cqt_target.append(Cqt_target)                
                                
            #.. update codebook (code book is updated after flag_buffer_update = True)
                if self.update_codebook and self.args.use_trj_Cqt == False:
                    self.vae.codebook_update(argmin, t_env, sum_rewards, reward_tgo[:,t]) # include for-batch          
        # end seq sampling loop ==============================================================================

        #.. mac / gprob ---
        #td = time.time()-ts
        #print( str(td))
        mac_out        = th.stack(mac_out, dim=1)            # Concat over time        
        if self.args.mac == "lagma_gc_mac" :
            goal_prob_logits_out  = th.stack(goal_prob_logits_out, dim=1)  # Concat over time [bs,t,n_agent,n_cluster]        

        if self.args.use_vqvae:            
            visit_nodes    = th.stack(visit_nodes, dim=1)  # Concat over time # sequence of trajectory
            th_state_input = th.stack(buf_state_input, dim=0 )
            th_recon       = th.stack(buf_recon      , dim=0 )
            th_z_e         = th.stack(buf_z_e        , dim=0 )
            th_latent_emb  = th.stack(buf_latent_emb , dim=0 )
            if self.args.recon_type == 2 or self.args.recon_type == 3:
                th_Cqt_hat     = th.stack(buf_Cqt_hat    , dim=0 )
                # th_Cqt_target  = th.stack(buf_Cqt_target , dim=0 )

        pred_accuracy  = th.zeros((self.bs*self.args.n_agents, self.args.max_seq_length ) ).to(self.args.device).detach()                        
        
        # 1. trajectory clustering--------------------------                
        if seq_labels is not None:
            #.. extend visit_nodes
            #ref_input_size = self.args.max_seq_length
            ref_input_size = f_classifier.fc1.in_features
            visit_sequences = self.default_node*th.ones((self.args.batch_size, self.args.max_seq_length) ).to(self.args.device).detach()
            visit_sequences[:,:visit_nodes.size(1)] = visit_nodes.detach() # [bs, t]
                
            indices_not_clustered = np.equal(seq_labels, -1)           
                
            #.. conduct classification
            if th.any(indices_not_clustered==True):
                if self.args.similarity_type <= 3:
                    if self.args.classifier_input_type==1: # full
                        input_seq = visit_sequences[indices_not_clustered==True,:]
                    else: # reduced
                        input_seq_raw = RBS_cluster.sequence_reduction( visit_sequences[indices_not_clustered==True,:] )
                        input_seq     = self.default_node*th.ones((input_seq_raw.size()[0], ref_input_size ) ).to(self.args.device).detach()                        
                        min_size      = min( input_seq_raw.size()[1], ref_input_size)
                        input_seq[:,:min_size] = input_seq_raw[:,:min_size]

                else: # use meanVQ
                    target_idx = np.nonzero(indices_not_clustered).squeeze(-1)
                    target_visited_sequence = visit_sequences[target_idx,:]
                    valid_idx = (target_visited_sequence != self.args.n_codes).cpu().numpy()
                    cur_qvectors = self.vae.emb.weight.permute(1,0).cpu().detach().numpy()
                    meanVq = []
                    for k in range(len(target_visited_sequence)):
                        ide = sum(valid_idx[k]).item()
                        if ide ==0: ide=1 # to prevent anomaly
                        if self.args.flag_reduced_seq: # here index order is ignored
                            ndx = np.array( list(set(target_visited_sequence[k,:ide].cpu().numpy().astype(np.int32)) ) )
                        else:
                            ndx = target_visited_sequence[k,:ide].cpu().numpy().astype(np.int32) 
                        
                        meanVq.append(np.mean(cur_qvectors[ndx,:], axis=0))
                    meanVq = np.array(meanVq)    
                    input_seq = th.tensor(meanVq)
                
                if f_classifier is not None:
                    outputs = f_classifier(input_seq[:,:ref_input_size].to(self.args.device))
                    _, pred_labels = th.max(outputs, 1)
                    seq_labels[indices_not_clustered==True] = pred_labels.to(th.int32).to('cpu')              
                else: # set default label
                    seq_labels[indices_not_clustered==True] = 0                

            seq_labels = seq_labels.unsqueeze(-1).unsqueeze(-1).expand((self.bs, batch.max_seq_length-1, self.args.n_agents)).to(th.long).to(self.args.device).detach()
                
            labels_reshaped = seq_labels.reshape(-1,1).squeeze(-1) 
                
            if self.args.mac == "lagma_gc_mac" :
                goal_prob_logits_out_reshaped = goal_prob_logits_out[:,:batch.max_seq_length-1].reshape(-1, self.args.n_cluster)
                mask_reshaped   =  th.nonzero(mask.expand_as(seq_labels).reshape(-1,1).squeeze(-1)).squeeze(-1)
        
                #.. only consider valid part ---                                         
                masked_prediction_loss = self.criterion(goal_prob_logits_out_reshaped[mask_reshaped], labels_reshaped[mask_reshaped])
                goal_pred_loss = masked_prediction_loss # .sum()/mask.sum() // averaging is already considered

                #.. for monitoring only                                        
                selected_goals_reshaped= selected_goals[:,:batch.max_seq_length-1].squeeze(-1).permute(0,2,1).reshape(-1, batch.max_seq_length-1).detach() # [bs,t,n_agent,1] --> [bs*n_agents,t_max]
                                        
                goal_label_vec = seq_labels.permute(0,2,1).reshape(-1, batch.max_seq_length-1).detach() # [bs*n_agents,t_max]
                TF_vec  = ( selected_goals_reshaped == goal_label_vec).float() # [bs*n_agents, t_max]
                pred_accuracy[:,:batch.max_seq_length-1] = TF_vec

            else:
                goal_pred_loss = th.tensor(0.0)
                
        else:
            goal_pred_loss = th.tensor(0.0)
                
        #-------------------------------------------------------------------------------------------------------
      
        # 2. VQVAE loss computation sampling loop =====================================================
        if self.args.use_vqvae:
            
            for t in range(batch.max_seq_length):
                #.. compute time&trajectory dependent indexing: ndx computation----------
                if self.args.use_trj_dependent_node:
                    if t == 0:
                        if self.args.ref_max_time == 1:
                            dn = (self.nk / batch.max_seq_length) # dn
                            dn_r = self.nk / batch.max_seq_length
                            # dr = self.nk % batch.max_seq_length
                            # ids = int(dn*batch.max_seq_length)
                            
                        elif self.args.ref_max_time == 2:
                            dn = (self.nk / self.args.max_seq_length ) # dn
                            dn_r = self.nk / self.args.max_seq_length
                            # dr = self.nk % self.args.max_seq_length
                            # ids = int(dn * self.args.max_seq_length)
                    
                    ndx_cluster = {}
                    max_len = 0
                    for k in range(self.args.n_cluster):
                        ids = self.nk * (k) + int(dn*t)
                        ide = self.nk * (k) + int(dn*(t+1))
                        if dn >= 1:       
                            # ndx = np.arange(dn*t, dn*(t+1), 1) 
                            ndx_cluster[k] = np.arange(ids, ide, 1)      
                            # if t < dr:
                            #     ndx[k] = np.append(ndx, np.array(ids+t))
                       
                        else:
                            #ids = int(self.nk * (k) + dn*t)
                            ndx_cluster[k] = np.array([ids])
                        max_len = max(max_len, len(ndx_cluster[k]))     
                        
                    if self.args.timestep_emb == False:
                        ndx = None
                    else:                        
                        ndx = self.default_node*np.ones((self.args.batch_size, max_len))
                        #ndx = self.default_node*np.ones(self.args.batch_size)
                        for kd in range(self.args.batch_size):
                            if seq_labels is None:
                                #.. consider all trajectory dependent VQ codebook here by randomly draw numbers
                                k_id = np.random.randint(0, self.args.n_cluster)
                                ndx[kd,:] = ndx_cluster[k_id]
                                #ndx[kd] = ndx_cluster[k_id]
                            else:
                                k_id = seq_labels[kd][0][0].item()
                                ndx[kd,:] = ndx_cluster[k_id]
                            
                #.. compute timedependent indexing only ----------
                else:
                    if t == 0:
                        if self.args.ref_max_time == 1:
                            dn = int(self.args.n_codes / batch.max_seq_length) # dn
                            dn_r = self.args.n_codes / batch.max_seq_length
                            dr = self.args.n_codes % batch.max_seq_length
                            ids = dn*batch.max_seq_length
                        elif self.args.ref_max_time == 2:
                            dn = int(self.args.n_codes / self.args.max_seq_length ) # dn
                            dn_r = self.args.n_codes / self.args.max_seq_length
                            dr = self.args.n_codes % self.args.max_seq_length
                            ids = dn * self.args.max_seq_length
                        
                    if dn >= 1:    
                        ndx = np.arange(dn*t, dn*(t+1), 1)      
                        if t < dr:
                            ndx = np.append(ndx, np.array(ids+t))
                       
                    else:
                        ndx = np.array([int(t*dn_r)])
                        
                    if self.args.timestep_emb == False:
                        ndx = None
                #--------------------------------------------------------------------------------

                #..matching current values
                state_input_t = th_state_input[t]
                recon_t       = th_recon[t]
                z_e_t         = th_z_e[t]         
                latent_emb_t  = th_latent_emb[t]
                argmin_t      = visit_nodes.permute((1,0))[t] # [B,t] --> [t,B]
                if self.args.recon_type ==2 or self.args.recon_type ==3:
                    Cqt_hat_t     = th_Cqt_hat[t]     
                    #Cqt_target_t  = th_Cqt_target[t]  
                    if self.args.use_trj_Cqt:
                        Cqt_target_t   = self.vae.call_Cqt_batch(argmin_t, seq_labels=seq_labels)
                    else:
                        Cqt_target_t   = self.vae.call_Cqt_batch(argmin_t)        

                #.. trajectory dependent Cqt update
                if self.update_codebook and self.args.use_trj_Cqt == True:
                    self.vae.codebook_update_tdvq(argmin_t, t_env, sum_rewards, reward_tgo[:,t], seq_labels=seq_labels) # include for-batch          

                #.. compute loss
                if self.args.flag_batch_wise_vqvae_loss:
                    if self.args.recon_type ==1:
                        vae_loss, ce_loss, vq_loss, commit_loss, coverage_loss = \
                            self.vae.loss_function_batch(state_input_t, recon_t, z_e_t, latent_emb_t, ndx=ndx)
                    elif self.args.recon_type ==2:
                        vae_loss, ce_loss, vq_loss, commit_loss, coverage_loss = \
                            self.vae.loss_function_batch(state_input_t, recon_t, z_e_t, latent_emb_t, ndx=ndx, Cqt=Cqt_target_t, recon_Cqt=Cqt_hat_t)                        
                    elif self.args.recon_type ==3:
                        vae_loss, ce_loss, vq_loss, commit_loss, coverage_loss = \
                            self.vae.loss_function_batch(state_input_t, recon_t, z_e_t, latent_emb_t, ndx=ndx, Cqt=Cqt_target_t, recon_Cqt=Cqt_hat_t)                        
                    vae_losses.append(vae_loss) 
                    ce_losses       += th.mean(ce_loss       )    
                    vq_losses       += th.mean(vq_loss       )
                    commit_losses   += th.mean(commit_loss   )
                    coverage_losses += th.mean(coverage_loss )
                else:
                    if self.args.recon_type ==1:
                        vae_loss, ce_loss, vq_loss, commit_loss, coverage_loss = \
                            self.vae.loss_function(state_input_t, recon_t, z_e_t, latent_emb_t, ndx=ndx)
                    elif self.args.recon_type ==2:
                        vae_loss, ce_loss, vq_loss, commit_loss, coverage_loss = \
                            self.vae.loss_function(state_input_t, recon_t, z_e_t, latent_emb_t, ndx=ndx, Cqt=Cqt_target_t, recon_Cqt=Cqt_hat_t)                        
                    elif self.args.recon_type ==3:
                        vae_loss, ce_loss, vq_loss, commit_loss, coverage_loss = \
                            self.vae.loss_function(state_input_t, recon_t, z_e_t, latent_emb_t, ndx=ndx, Cqt=Cqt_target_t, recon_Cqt=Cqt_hat_t)                        
                        
                    # this results are already computed by taking average in batch-wise
                    vae_losses      += vae_loss 
                    ce_losses       += ce_loss
                    vq_losses       += vq_loss
                    commit_losses   += commit_loss
                    coverage_losses += coverage_loss
        # end sampling loop ==============================================================================
           
        if self.update_vqvae: 
            if self.args.flag_batch_wise_vqvae_loss:
                vae_losses      = th.stack(vae_losses, dim=1) # [bs, t]
                vae_losses      = vae_losses[:,:-1].unsqueeze(-1)
            else:
                vae_losses      /= batch.max_seq_length # compute average by timestep
            ce_losses       /= batch.max_seq_length
            vq_losses       /= batch.max_seq_length
            commit_losses   /= batch.max_seq_length
            coverage_losses /= batch.max_seq_length

            self.ce_losses       = ce_losses       
            self.vq_losses       = vq_losses       
            self.commit_losses   = commit_losses   
            self.coverage_losses = coverage_losses 
        else: # for monitoring
            vae_losses        = self.vae_losses     
        # End vqvae training =============================================================================

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim
           
        if self.args.mac == "lagma_gc_mac" :
            goal_prob_logits_out = goal_prob_logits_out[:, :-1] # [bs, t, n_agent, n_cluster]
        #chosen_goal_prob    = th.gather(goal_prob_out[:, :-1], dim=3, index=selected_goals.long()).squeeze(3)

        if self.args.mixer == "dmaq_qatten":
            x_mac_out = mac_out.clone().detach()
            x_mac_out[avail_actions == 0] = -9999999
            max_action_qvals, max_action_index = x_mac_out[:, :-1].max(dim=3)
            max_action_index = max_action_index.detach().unsqueeze(3)
            is_max_action = (max_action_index == actions).int().float()
                
        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            if self.args.mac == "lagma_gc_mac":                
                target_agent_outs, _, _  = self.target_mac.forward(batch, t=t, goal_latent=None, selected_goal=selected_goals[:,t].squeeze(-1).reshape(-1))            
            else:
                target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)
        
        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q:            
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            
            if self.args.mixer == "dmaq_qatten":
                target_chosen_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
                target_max_qvals = target_mac_out.max(dim=3)[0]
                target_next_actions = cur_max_actions.detach()

                cur_max_actions_onehot = to_cuda(th.zeros(cur_max_actions.squeeze(3).shape + (self.args.n_actions,)), self.args.device)
                cur_max_actions_onehot = cur_max_actions_onehot.scatter_(3, cur_max_actions, 1)
            else:                
                target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)            
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        if self.mixer is not None:
            if self.args.mixer == "dmaq_qatten":
                ans_chosen, q_attend_regs, head_entropies = \
                    self.mixer(chosen_action_qvals, batch["state"][:, :-1], is_v=True)
                ans_adv, _, _ = self.mixer(chosen_action_qvals, batch["state"][:, :-1], actions=actions_onehot,
                                      max_q_i=max_action_qvals, is_v=False)
                chosen_action_qvals = ans_chosen + ans_adv
                
                if self.args.double_q:                
                    target_chosen, _, _ = self.target_mixer(target_chosen_qvals, batch["state"][:, 1:], is_v=True)
                    target_adv, _, _ = self.target_mixer(target_chosen_qvals, batch["state"][:, 1:],
                                                         actions=cur_max_actions_onehot,
                                                         max_q_i=target_max_qvals, is_v=False)
                    target_max_qvals = target_chosen + target_adv
                
            else: # vdn & qmix
                chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
                target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])
            

        #td = time.time()-ts
        #print( str(td))
        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals
        
        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask        

        # Normal L2 loss, take mean over actual data         
        if self.args.mixer == "dmaq_qatten":        
            loss = (masked_td_error ** 2).sum() / mask.sum()  + q_attend_regs \
                + self.args.lambda_gp_loss * goal_pred_loss
                      
        else:
            loss = (masked_td_error ** 2).sum() / mask.sum() \
                + self.args.lambda_gp_loss * goal_pred_loss
            
        # Optimise
        #.. policy learning
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        #.. VQ-VAE learning
        if self.args.use_vqvae and self.update_vqvae:            
            self.vae_optimizer.zero_grad()
            grad_norm = th.nn.utils.clip_grad_norm_(self.vae_params, self.args.grad_norm_clip)
            
            vae_losses.backward()
            self.vae_losses = vae_losses
            
            self.vae_optimizer.step()  

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            mask_elems = mask.sum().item()

            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("gp_loss", goal_pred_loss.item(), t_env)

            if self.args.use_vqvae:
                self.logger.log_stat("vae_loss", self.vae_losses.item(), t_env)                    

                self.logger.log_stat("ce_loss", self.ce_losses.item(), t_env)
                self.logger.log_stat("vq_loss", self.vq_losses.item(), t_env)
                self.logger.log_stat("commit_loss", self.commit_losses.item(), t_env)
                self.logger.log_stat("coverage_loss", self.coverage_losses.item(), t_env)

            self.logger.log_stat("grad_norm", grad_norm, t_env)
            
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env                       
        
        if self.args.flag_seq_cluster:
            # zero-pad sequence
            visit_nodes_padded = self.args.n_codes*th.ones(visit_nodes.size()[0], self.args.max_seq_length)
            visit_nodes_padded[:,:batch.max_seq_length] = visit_nodes
        else:
            visit_nodes_padded = None
            
        return visit_nodes_padded, mean_acc_t0 # cpu

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")
                
    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.args.use_vqvae:
            self.vae.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)        
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))
        if self.args.use_vqvae:
            th.save(self.vae.state_dict(), "{}/vae.th".format(path))
            th.save(self.vae.emb.state_dict(), "{}/codebook.th".format(path))
            # additional codebook infomation
            if self.args.save_vae_info:
                self.vae.save_vae_info(path) 
        
    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.args.n_agents

        return input_shape

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
        if self.args.use_vqvae:
            self.vae.load_state_dict(th.load("{}/vae.th".format(path), map_location=lambda storage, loc: storage))
            self.vae.emb.load_state_dict(th.load("{}/codebook.th".format(path), map_location=lambda storage, loc: storage))