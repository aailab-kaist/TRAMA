from re import I
import torch as th 
import torch.nn as nn
import torch.nn.functional as F
import math
from yaml import safe_dump_all
from torch.distributions import Categorical
from .nearest_embed import NearestEmbed, NearestEmbedEMA
import heapq
import os
import numpy as np
import pickle

class TopKHeap:
    def __init__(self, k, args):
        self.args = args
        self.k = k
        self.Cq0 = [] # at most k-values        
        self.seq = {} # at most k-sequences
        self.des = [] # at most k-values     
        self.default_class = 0 # set the first class as default class
        
    def sample_seq(self):
        heap_size = len(self.Cq0)
        
        if self.args.sampling_type == 1: # random sampling
            idx = th.randint(0,heap_size,size=(1,1)).item()
            
        elif self.args.sampling_type == 2: # softmax sampling
            idx = Categorical(th.stack(self.Cq0,dim=0).to(self.args.device)).sample().long().item()

        return self.Cq0[idx], self.seq[idx], self.des[idx]
    
    def CanSample(self): 
        if len(self.Cq0) > 0:
            return True
        else:
            return False
        
    def CanSample_backup(self, trj_idx=None):

        if self.args.use_trj_Cqt:
            if trj_idx is not None:
                len_ref = len(self.Cq0[trj_idx])
            else:
                len_ref = len(self.Cq0[self.default_class]) 
        else:
            len_ref = len(self.Cq0)    

        if len(len_ref) > 0:
            return True
        else:
            return False
        
    
    def push(self, Cq0, seq, des):        
        if len(self.Cq0) < self.k:
            self.heappush(self.Cq0, self.seq, self.des, Cq0, seq, des)
        else:
            min_val = self.Cq0[0]
            #if Cq0 > min_val:
            if Cq0 >= min_val: # if the value is same, then replace the value with recent data
                self.heapreplace(self.Cq0, self.seq, self.des, Cq0, seq, des)
                
    def heappush(self, heap, heap_seq, heap_des, x, seq, des):
        idx = len(heap)
        heap.append(x) # append latest data 
        heap_seq[idx] = seq 
        heap_des.append(des)
        self.siftdown(heap, heap_seq, heap_des, 0, len(heap) - 1)

    def heapreplace(self, heap, heap_seq, heap_des, x, seq, des):
        if heap:
            #removed = heap[0]
            heap[0] = x
            heap_seq[0] = seq
            heap_des[0] = des
            self.siftup(heap, heap_seq, heap_des, 0)
            #return removed
        else:
            raise IndexError("heap is empty")

    def siftdown(self, heap, heap_seq, heap_des, startpos, pos):
        newitem     = heap[pos]
        newitem_seq = heap_seq[pos]
        newitem_des = heap_des[pos]
        
        while pos > startpos:            
            parentpos  = pos - 1
            parent     = heap[parentpos]
            parent_seq = heap_seq[parentpos]
            parent_des = heap_des[parentpos]
            
            if newitem < parent:
                heap[pos] = parent
                heap_seq[pos] = parent_seq
                heap_des[pos] = parent_des
                pos = parentpos
            else:
                break
            
        # is this effective?    
        heap[pos]     = newitem     # newitem's right position
        heap_seq[pos] = newitem_seq # newitem's right position
        heap_des[pos] = newitem_des # newitem's right position
                
    def siftup(self, heap, heap_seq, heap_des, pos):
        endpos = len(heap)
        #startpos = pos
        newitem = heap[pos]
        newitem_seq = heap_seq[pos]
        newitem_des = heap_des[pos]
        
        childpos = pos + 1
        
        while childpos < endpos:            
            child = heap[childpos]
            child_seq = heap_seq[childpos]
            child_des = heap_des[childpos]
            if child < newitem:
                heap[pos]     = child
                heap_seq[pos] = child_seq
                heap_des[pos] = child_des
            else:
                break
            pos = childpos
            childpos = pos + 1
            
        heap[pos]     = newitem
        heap_seq[pos] = newitem_seq        
        heap_des[pos] = newitem_des        
        
        #self.siftdown(heap, heap_seq, startpos, pos) # needed?        
        #check_result =1

class VQVAE(nn.Module):
    def __init__(self, input_shape, state_dim, args):
        super(VQVAE, self).__init__()
        self.args = args
        self.state_dim = state_dim
        self.buffer_update_time = args.buffer_update_time
        self.flag_buffer_update = False
        
        #.. add additional code for zero-state vector
        if self.args.flag_zero_state_management:
            total_n_codes = args.n_codes + 1
        else:
            total_n_codes = args.n_codes

        #..code book generation ---------
        #self.Ncall = th.zeros(args.n_codes,1, dtype=int).to(self.args.device)
        self.Ncall    = th.ones (total_n_codes, dtype=int).to(self.args.device)
        self.Rq0      = th.zeros(total_n_codes).to(self.args.device)
        self.Rqt      = th.zeros(total_n_codes).to(self.args.device)
        self.timestep = th.zeros(total_n_codes).to(self.args.device)
        
        #.. for recursive update formula
        # self.prob_param_0 = th.zeros(args.n_codes,2).to(self.args.device) # mu, sigma
        # self.prob_param_t = th.zeros(args.n_codes,2).to(self.args.device) # mu, sigma
        
        #.. to compute moving average/variance                
        if self.args.use_trj_Cqt:
            self.curr_capacity = th.ones (total_n_codes, self.args.n_cluster, dtype=int).to(self.args.device)
            self.Buffer_R0     = th.zeros(total_n_codes, self.args.n_cluster, args.n_max_code).to(self.args.device)
            self.Buffer_Rt     = th.zeros(total_n_codes, self.args.n_cluster, args.n_max_code).to(self.args.device)
        else:
            self.curr_capacity = th.ones (total_n_codes, dtype=int).to(self.args.device)
            self.Buffer_R0     = th.zeros(total_n_codes, args.n_max_code).to(self.args.device)
            self.Buffer_Rt     = th.zeros(total_n_codes, args.n_max_code).to(self.args.device)

        #.. to keep trajectory sequence        
        #self.seq_capacity = th.zeros(args.n_codes, dtype=int).to(self.args.device)
        #self.Buffer_seq = th.zeros(args.n_codes, args.k_top_seq, args.max_seq_length).to(self.args.device)
        #self.Buffer_Cq0 = th.zeros(args.n_codes, args.k_top_seq).to(self.args.device)
        
        self.Buffer_seq = {}        
        
        if self.args.use_trj_Cqt:
            for i in range(total_n_codes):
                self.Buffer_seq[i] = {}
                for j in range(self.args.n_cluster):
                    self.Buffer_seq[i][j] = TopKHeap(args.k_top_seq, args)   
        else:
            for i in range(total_n_codes):
                self.Buffer_seq[i] = TopKHeap(args.k_top_seq, args)   

        # -------------------------------
        if args.recon_type == 3: # dCAE structure
            encode_input_shape = input_shape + 1 # +1 for timestep input
        else:
            encode_input_shape = input_shape
            
        # self.fc0 = nn.Linear(encode_input_shape, args.vae_hidden_dim)
        # self.fc1 = nn.Linear(args.vae_hidden_dim, args.vae_hidden_dim)
        # self.fc2 = nn.Linear(args.vae_hidden_dim, args.latent_dim)

        self.emb = NearestEmbed(total_n_codes, args.latent_dim)
        
        if args.recon_type == 3: # dCAE structure
            recon_input_dim = args.latent_dim + 1 # +1 for timestep input
            self.fc3 = nn.Linear(recon_input_dim, args.vae_hidden_dim)
        else:
            self.fc3 = nn.Linear(args.latent_dim, args.vae_hidden_dim)
            
        #self.fc4 = nn.Linear(args.n_agents * args.vae_hidden_dim, args.vae_hidden_dim) # modified
        self.fc4 = nn.Linear(args.vae_hidden_dim, args.vae_hidden_dim)
        if self.args.recon_type == 1:   # state reconstruction
            self.fc5 = nn.Linear(args.vae_hidden_dim, state_dim) 
        elif self.args.recon_type == 2: # value estimation 
            self.fc5_S   = nn.Linear(args.vae_hidden_dim, state_dim) 
            self.fc5_Cqt = nn.Linear(args.vae_hidden_dim, 1) 
        elif self.args.recon_type == 3: # both state/value estimation            
            self.fc5_S   = nn.Linear(args.vae_hidden_dim, state_dim) 
            self.fc5_Cqt = nn.Linear(args.vae_hidden_dim, 1) 
        
        self.state_embed_net = nn.Sequential(nn.Linear(encode_input_shape, args.vae_hidden_dim),
                                            nn.ReLU(),
                                            nn.Linear(args.vae_hidden_dim, args.vae_hidden_dim ),
                                            nn.ReLU(),                                            
                                            nn.Linear(args.vae_hidden_dim, args.latent_dim )).to(self.args.device)
        

        self.ce_loss       = th.zeros(1).to(self.args.device)
        self.mse           = th.zeros(1).to(self.args.device)
        self.vq_loss       = th.zeros(1).to(self.args.device)
        self.commit_loss   = th.zeros(1).to(self.args.device)
        self.coverage_loss = th.zeros(1).to(self.args.device)
        
        #.. what for?
        # self.state_embed_net[-1].weight.detach().fill_(1 / 40)

        # self.emb.weight.detach().normal_(0, 0.02)
        # th.fmod(self.emb.weight, 0.04)

    def encode(self, inputs):
        #return self.fc2(F.relu(self.fc1(F.relu(self.fc0(inputs)))))
        return self.state_embed_net(inputs)
    
    def decode(self, z):
        #rec = F.relu(self.fc3(z)).view(-1, self.args.n_agents * self.args.vae_hidden_dim) # modified
        rec_hidden = F.relu(self.fc3(z)).view(-1, self.args.vae_hidden_dim) 
        
        if self.args.recon_type == 2 or self.args.recon_type == 3:
            rec     = self.fc5_S(F.relu(self.fc4(rec_hidden)))
            Cqt_hat = self.fc5_Cqt(F.relu(self.fc4(rec_hidden)))
            return rec, Cqt_hat
        else:
            rec = self.fc5(F.relu(self.fc4(rec_hidden)))
            return rec

    def forward(self, inputs, timestep=None ):
        z_e = self.encode(inputs)
        #z_q, _ = self.emb(z_e, weight_sg=True) # choose the nearest embedding
        z_q, argmin = self.emb(z_e, weight_sg=True) # choose the nearest embedding
        emb, _ = self.emb(z_e.detach())
        # preserve gradients
        # z_q = z_e + (z_q - z_e).detach()
        # role_emb, _ = self.emb(z_e.detach())

        if self.args.recon_type == 3 and timestep is not None:
            decoder_input = th.cat( [z_q, timestep], dim=1) # [bs,dim]
            s_hat, Cqt_hat = self.decode(decoder_input)
            return s_hat, z_e, emb, argmin, Cqt_hat
        
        else:
            if self.args.recon_type == 2:
                s_hat, Cqt_hat = self.decode(z_q)
                return s_hat, z_e, emb, argmin, Cqt_hat    
            else:
                s_hat = self.decode(z_q)            
                return s_hat, z_e, emb, argmin
    
    def codebook_update(self, argmin, t_env, sum_rewards, reward_tgo):
        
        if self.flag_buffer_update == False and t_env >= self.buffer_update_time:
            self.flag_buffer_update = True
            
        n_batch_size = sum_rewards.size()[0]

        if self.flag_buffer_update:             
            #.. update parameters # what if argmin is overlapped ?
            # self.prob_param_0[argmin,1] += (th.mul( self.Ncall[argmin], sum_rewards) - self.prob_param_0[argmin,0])**2/(th.mul(self.Ncall[argmin],self.Ncall[argmin]+1)) 
            # self.prob_param_0[argmin,0] += sum_rewards
            
            # self.prob_param_t[argmin,1] += (th.mul( self.Ncall[argmin], reward_tgo) - self.prob_param_t[argmin,0])**2/(th.mul(self.Ncall[argmin],self.Ncall[argmin]+1)) 
            # self.prob_param_t[argmin,0] += reward_tgo
            self.Ncall[argmin] += 1 
            
            #.. update buffer to compute moving statistics (require batch-loop operation)
            #for k in range(self.args.batch_size):
            for k in range(n_batch_size):
                idx = argmin[k]
                if self.curr_capacity[idx] >= self.args.n_max_code:
                    # FIFO structure
                    self.Buffer_R0[idx,:-1] = self.Buffer_R0[idx,1:].clone()
                    self.Buffer_Rt[idx,:-1] = self.Buffer_Rt[idx,1:].clone()
                    
                    self.Buffer_R0[idx,-1] = sum_rewards[k]           
                    self.Buffer_Rt[idx,-1] = reward_tgo[k]
                else:
                    self.Buffer_R0[idx,self.curr_capacity[idx]] = sum_rewards[k]           
                    self.Buffer_Rt[idx,self.curr_capacity[idx]] = reward_tgo[k]
                    self.curr_capacity[idx] += 1
                    
        return 0   
    
    def codebook_update_tdvq(self, argmin, t_env, sum_rewards, reward_tgo, seq_labels=None):
        
        if self.flag_buffer_update == False and t_env >= self.buffer_update_time:
            self.flag_buffer_update = True
            
        n_batch_size = sum_rewards.size()[0]

        if self.flag_buffer_update:       
            self.Ncall[argmin] += 1
            
            #.. update buffer to compute moving statistics (require batch-loop operation)
            #for k in range(self.args.batch_size):
            for k in range(n_batch_size):
                idx = argmin[k]
                if seq_labels is None:
                    jdx = np.arange(0, self.args.n_cluster)
                    if self.curr_capacity[idx][0] >= self.args.n_max_code:
                        # FIFO structure
                        self.Buffer_R0[idx,jdx,:-1] = self.Buffer_R0[idx,jdx,1:].clone()
                        self.Buffer_Rt[idx,jdx,:-1] = self.Buffer_Rt[idx,jdx,1:].clone()
                    
                        self.Buffer_R0[idx,jdx,-1] = sum_rewards[k]           
                        self.Buffer_Rt[idx,jdx,-1] = reward_tgo[k]
                    else:
                        self.Buffer_R0[idx,jdx,self.curr_capacity[idx][jdx]] = sum_rewards[k]           
                        self.Buffer_Rt[idx,jdx,self.curr_capacity[idx][jdx]] = reward_tgo[k]
                        self.curr_capacity[idx][jdx] += 1
                else:
                    jdx = seq_labels[k][0][0].item()
                
                    if self.curr_capacity[idx][jdx] >= self.args.n_max_code:
                        # FIFO structure
                        self.Buffer_R0[idx,jdx,:-1] = self.Buffer_R0[idx,jdx,1:].clone()
                        self.Buffer_Rt[idx,jdx,:-1] = self.Buffer_Rt[idx,jdx,1:].clone()
                    
                        self.Buffer_R0[idx,jdx,-1] = sum_rewards[k]           
                        self.Buffer_Rt[idx,jdx,-1] = reward_tgo[k]
                    else:
                        self.Buffer_R0[idx,jdx,self.curr_capacity[idx][jdx]] = sum_rewards[k]           
                        self.Buffer_Rt[idx,jdx,self.curr_capacity[idx][jdx]] = reward_tgo[k]
                        self.curr_capacity[idx][jdx] += 1
                    
        return 0    
    
    def save_vae_info(self, savepath):
        np.save(os.path.join(savepath, 'Ncall')    , np.array(self.Ncall.cpu())     )
        np.save(os.path.join(savepath, 'Buffer_R0'), np.array(self.Buffer_R0.cpu()) )
        np.save(os.path.join(savepath, 'Buffer_Rt'), np.array(self.Buffer_Rt.cpu()) )
        save_file_path = os.path.join(savepath, 'Buffer_Seq.pickle')
        with open(save_file_path, 'wb') as pickle_file:
            pickle.dump(self.Buffer_seq, pickle_file)
    
    def call_statistics(self, argvec):
        # argvec: index vectors
        mu_0  = th.mean( self.Buffer_R0[argvec,:self.curr_capacity[argvec]] )
        std_0 = th.std(  self.Buffer_R0[argvec,:self.curr_capacity[argvec]] )
        
        mu_t  = th.mean( self.Buffer_Rt[argvec,:self.curr_capacity[argvec]] )
        std_t = th.std(  self.Buffer_Rt[argvec,:self.curr_capacity[argvec]] )
        
        return mu_0, std_0, mu_t, std_t
    
    def call_Cq0(self, argvec):
        mu_0  = th.mean( self.Buffer_R0[argvec,:self.curr_capacity[argvec]] )
        std_0 = th.std(  self.Buffer_R0[argvec,:self.curr_capacity[argvec]] )

        Cq0 = mu_0 + self.args.lambda_exp*std_0         

        return Cq0
    
    def call_Cqt(self, argvec):
        mu_t  = th.mean( self.Buffer_Rt[argvec,:self.curr_capacity[argvec]] )
        std_t = th.std(  self.Buffer_Rt[argvec,:self.curr_capacity[argvec]] )

        #Cqt = mu_t + self.args.lambda_exp*std_t
        if self.args.flag_UCB_incentive:
            Cqt = mu_t + 4*std_t*math.pow(math.log(self.args.UCB_param_t)/self.Ncall[argvec], 0.5)
        else:
            Cqt = mu_t

        return Cqt
    
    def call_Cqt_tdvq(self, argvec, id_batch, trj_idx=None):

        if trj_idx is None:
            jdx = int(0) # call node value of the first clas
        else:
            jdx = trj_idx[id_batch][0][0].item()
            
        # input: argvec - single value 
        mu_t  = th.mean( self.Buffer_Rt[argvec, jdx, :self.curr_capacity[argvec,jdx]] )
        std_t = th.std(  self.Buffer_Rt[argvec, jdx, :self.curr_capacity[argvec,jdx]] )

        #Cqt = mu_t + self.args.lambda_exp*std_t
        if self.args.flag_UCB_incentive:
            Cqt = mu_t + 4*std_t*math.pow(math.log(self.args.UCB_param_t)/self.Ncall[argvec], 0.5)
        else:
            Cqt = mu_t

        return Cqt
    
    def call_Cqt_batch(self, bat_argvec, seq_labels=None):
        n_batch_size = bat_argvec.size()[0]
        Cqt_out = []
        #for i in range(self.args.batch_size):
        for i in range(n_batch_size):
            if self.args.use_trj_Cqt:      
                Cqt = self.call_Cqt_tdvq(bat_argvec[i] , i , trj_idx=seq_labels )
            else:
                Cqt = self.call_Cqt(bat_argvec[i])
            Cqt_out.append(Cqt)
        Cqt_out = th.stack(Cqt_out, dim=0).detach()
        return Cqt_out

    def call_emb(self):
        return self.emb.call_emb()

    def f_coverage_loss(self, z_e):        
        coverage_loss = th.mean(th.norm( (z_e.unsqueeze(-1).detach()-self.emb.call_emb() )**2,2,1)).detach()
        return coverage_loss

    def loss_function(self, s, recon_s, z_e, emb_cur, ndx = None, Cqt=None, recon_Cqt=None):
        # z_e, emb_cur: [bs,dim]        
        
        # self.vq_loss     = F.mse_loss(emb, z_e.detach())
        # self.commit_loss = F.mse_loss(z_e, emb.detach())

        if (self.args.recon_type == 2 or self.args.recon_type == 3) and Cqt is not None and recon_Cqt is not None:
            self.ce_loss = self.args.recon_coef*F.mse_loss(recon_s, s) + F.mse_loss(recon_Cqt.squeeze(-1), Cqt.detach())
        else:
            self.ce_loss = F.mse_loss(recon_s, s)
        
        if self.args.flag_loss_type == 2: # MSE loss
            self.vq_loss     = F.mse_loss(z_e.detach(), emb_cur) # th.mean(th.norm((emb_cur - z_e.detach())**2, 2, 1))
            self.commit_loss = F.mse_loss(z_e, emb_cur.detach()) # th.mean(th.norm((emb_cur.detach() - z_e)**2, 2, 1))                

            if ndx is not None: # only update ndx embedding # need modification    
                if self.args.use_trj_dependent_node:
                    vq_vectors = self.emb.weight[:,ndx]   
                else:
                    vq_vectors = self.emb.weight[:,ndx]  
                    
                self.coverage_loss = F.mse_loss(z_e.unsqueeze(-1).detach(), vq_vectors )
            else:                
                self.coverage_loss = F.mse_loss(z_e.unsqueeze(-1).detach(), self.emb.weight )
                
        elif self.args.flag_loss_type == 1: # L2 norm
            self.vq_loss     = th.mean(th.norm((emb_cur - z_e.detach()), 2, 1)**2 )
            self.commit_loss = th.mean(th.norm((emb_cur.detach() - z_e), 2, 1)**2 )  
            
            if ndx is not None: # only update ndx embedding (note ndx.size doesn't have to be equal to batch size)     
                if self.args.use_trj_dependent_node:
                    # z_e: [bs, latent_dim], vq_vectors [latent_dim, bs, refer_size]    
                    vq_vectors = (self.emb.weight[:,ndx]).permute(1,0,-1)   
                    self.coverage_loss = th.mean(th.norm( (z_e.unsqueeze(-1).detach()-vq_vectors ),2,1)**2 )
                else:
                    # z_e: [bs, latent_dim], vq_vectors [latent_dim, refer_size]    
                    vq_vectors = self.emb.weight[:,ndx]   
                    self.coverage_loss = th.mean(th.norm( (z_e.unsqueeze(-1).detach()-vq_vectors ),2,1)**2 )
            else:
                self.coverage_loss = th.mean(th.norm( (z_e.unsqueeze(-1).detach()-self.emb.weight ),2,1)**2 )
                
        else: # original    
            self.vq_loss     = th.mean(th.norm((emb_cur - z_e.detach())**2, 2, 1) )
            self.commit_loss = th.mean(th.norm((emb_cur.detach() - z_e)**2, 2, 1) )  
            if ndx is not None: # only update ndx embedding            
                vq_vectors = self.emb.weight[:,ndx]            
                self.coverage_loss = th.mean(th.norm( (z_e.unsqueeze(-1).detach()-vq_vectors )**2,2,1) )
            else:
                self.coverage_loss = th.mean(th.norm( (z_e.unsqueeze(-1).detach()-self.emb.weight )**2,2,1) )
                

        total_loss = self.args.ce_coef * self.ce_loss + self.args.vq_coef * self.vq_loss \
            + self.args.commit_coef * self.commit_loss + self.args.coverage_coef * self.coverage_loss         

        return total_loss, self.ce_loss.detach(), self.vq_loss.detach(), self.commit_loss.detach(), self.coverage_loss.detach()
    
    def loss_function_batch(self, s, recon_s, z_e, emb_cur, ndx = None, Cqt=None, recon_Cqt=None):
        # z_e, emb_cur: [bs,dim]        
        
        # self.vq_loss = F.mse_loss(emb, z_e.detach())
        # self.commit_loss = F.mse_loss(z_e, emb.detach())

        if (self.args.recon_type == 2 or self.args.recon_type == 3) and Cqt is not None and recon_Cqt is not None:
            self.ce_loss = self.args.recon_coef*F.mse_loss(recon_s, s) + F.mse_loss(recon_Cqt, Cqt.detach())
            
            #self.ce_loss = self.args.recon_coef*th.norm((recon_s - s.detach())**2,2,1) + th.norm((recon_Cqt - Cqt.detach())**2,2,1)
        else:
            self.ce_loss = th.mean((recon_s-s)**2, dim=1)
            #self.ce_loss = th.norm((recon_s - s.detach())**2,2,1)
            
        # self.vq_loss     = (th.norm((emb_cur - z_e.detach())**2, 2, 1))
        # self.commit_loss = (th.norm((emb_cur.detach() - z_e)**2, 2, 1))   
        self.vq_loss     = th.mean( (emb_cur-z_e.detach())**2, dim=1) 
        self.commit_loss = th.mean( (emb_cur.detach()-z_e)**2, dim=1)            
    
        #self.coverage_loss = th.mean(th.norm( (z_e.unsqueeze(-1).detach()-self.emb.call_emb() )**2,2,1))

        if ndx is not None: # only update ndx embedding            
            vq_vectors = self.emb.weight[:,ndx]            
            #self.coverage_loss = th.mean((th.norm( (z_e.unsqueeze(-1).detach()-vq_vectors )**2,2,1)), dim=1)
            self.coverage_loss = th.mean(th.mean( (z_e.unsqueeze(-1).detach()-vq_vectors )**2 , dim=2), dim=1)
        else:
            #self.coverage_loss = th.mean((th.norm( (z_e.unsqueeze(-1).detach()-self.emb.weight )**2,2,1)), dim=1)
            self.coverage_loss = th.mean(th.mean( (z_e.unsqueeze(-1).detach()-self.emb.weight )**2 , dim=2), dim=1)

        total_loss = self.args.ce_coef * self.ce_loss + self.args.vq_coef * self.vq_loss + \
        self.args.commit_coef * self.commit_loss + self.args.coverage_coef * self.coverage_loss         

        return total_loss, self.ce_loss.detach(), self.vq_loss.detach(), self.commit_loss.detach(), self.coverage_loss.detach()
    
