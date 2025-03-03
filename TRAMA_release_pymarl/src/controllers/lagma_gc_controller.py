from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
from components.goal_selectors import REGISTRY as goal_REGISTRY
import torch as th
from torch import nn
import torch.optim as optim

# This multi-agent controller shares parameters between agents
class LAGMAMAC_GC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        #self.n_goals    = args.n_max_code
        self.latent_dim = args.latent_dim
        self.bs = args.batch_size
        self.args = args
        self.n_clusters    = args.n_cluster
        self.n_goals       = self.n_clusters 
        self.goal_repr_dim = args.latent_dim
        self.selected_goal_index = None
        #self.true_goal = -1*th.ones(args.n_agents,dtype=th.int32)
        self.true_goal = None
        input_shape = self._get_input_shape(scheme)
        
        # default: one-hot vector
        self.goal_onehot     = th.eye(self.n_clusters).to(device=args.device) # [n_goal, n_goal]
        
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)
        
        self.goal_selector           = goal_REGISTRY[args.goal_selector](args)       
                                
        self.hidden_states = None
        self.goal_hidden_states = None       
        
        self.softmax = nn.Softmax(dim=1) # Add softmax layer
        # for initialization, this will be replaced by VQVAE's embedding
        #self.goal_latent = nn.Parameter(th.rand(self.args.latent_dim, self.args.n_max_code )) # emb~U[0,1)
        
    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), goal_latent=None, test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        
        #.. action selection without goal 
        #agent_outputs = self.forward(ep_batch, t_ep, goal_latent=goal_latent, test_mode=test_mode)
        #chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        
        #.. action selection with goal selection first
        agent_outputs, _, _ = self.forward(ep_batch, t_ep, goal_latent=None, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)

        return chosen_actions, self.selected_goal_index

    def forward(self, ep_batch, t, goal_latent=None, test_mode=False, selected_goal=None):        
        selected_goal_latent = None        
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t] # only used for "pi_logits"
        
        #.. step 1: select goal based on local obsevation        
        goal_probs_logit = None        
        goal_probs_logit, self.goal_hidden_states = self.goal_predictor(agent_inputs, self.goal_hidden_states )
        
        #.. step 2: selected goal-onehot
        if selected_goal is None:
            if (t % self.args.n_pred_step) == 0:
                self.selected_goal_index, _ = self.goal_selector.select_goal(goal_probs_logit, test_mode=test_mode)            
        else:
            self.selected_goal_index = selected_goal
        
        selected_goal_input = self.goal_onehot[self.selected_goal_index] # tensor: [bs*n,n_cluster]
        
        #.. step 3: action selection conditioned on selected goal
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states, selected_goal_input)

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":
            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10

            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
            if not test_mode:
                # Epsilon floor
                epsilon_action_num = agent_outs.size(-1)
                if getattr(self.args, "mask_before_softmax", True):
                    # With probability epsilon, we will pick an available action uniformly
                    epsilon_action_num = reshaped_avail_actions.sum(dim=1, keepdim=True).float()

                agent_outs = ((1 - self.action_selector.epsilon) * agent_outs
                               + th.ones_like(agent_outs) * self.action_selector.epsilon/epsilon_action_num)

                if getattr(self.args, "mask_before_softmax", True):
                    # Zero out the unavailable actions
                    agent_outs[reshaped_avail_actions == 0] = 0.0

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1), \
            (None if goal_probs_logit is None else goal_probs_logit.view(ep_batch.batch_size, self.n_agents, -1)), \
                (None if selected_goal_latent is None else selected_goal_latent.view(ep_batch.batch_size, self.n_agents, -1))                    

    def update_goal_predictor(self, goal_prob_logits_out_reshaped, labels_reshaped ,mask_reshaped ):
        
        masked_prediction_loss = self.criterion(goal_prob_logits_out_reshaped[mask_reshaped], labels_reshaped[mask_reshaped])
        
        self.optimizer.zero_grad()
        masked_prediction_loss.backward()
        th.nn.utils.clip_grad_norm_(self.goal_predictor_params, 1.)
        self.optimizer.step
        
        return masked_prediction_loss.detach() # for logging
    
    def target_update_goal_predictor(self):
        self.target_goal_predictor.load_state_dict(
            self.goal_predictor.state_dict() )

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav
        self.goal_hidden_states        = self.goal_predictor.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav
        
    def parameters(self):
        #return self.agent.parameters()
        params  = list(self.agent.parameters())        
        params += list(self.goal_predictor.parameters()) # note that selector does not have any parameters
        return params
    
    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())
        self.goal_predictor.load_state_dict(other_mac.goal_predictor.state_dict())
        
    def cuda(self):
        self.agent.cuda()
        self.goal_predictor.cuda()
        
    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))
        th.save(self.goal_predictor.state_dict(), "{}/goal_predictor.th".format(path))
        
    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))
        self.goal_predictor.load_state_dict(th.load("{}/goal_predictor.th".format(path), map_location=lambda storage, loc: storage))
        
    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)
        self.goal_predictor = agent_REGISTRY[self.args.gp_agent](input_shape, self.args)                
        
    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape
    
    def select_action_only(self, ep_batch, t, goal_latent=None, selected_goal=None):
        # used only during training
        if goal_latent is not None:
            goal_latent = goal_latent.permute(1,0)

        agent_inputs = self._build_inputs(ep_batch, t)
        
        # [bs*n_agent, n_code, latent]
        goal_latent_exp = goal_latent.unsqueeze(0).repeat(agent_inputs.shape[0], 1, 1)
        
        #.. random sampling [bs*n_agent]
        if selected_goal is None:            
            selected_goal_index = th.randint(0, self.n_goals, (agent_inputs.shape[0],) ).to(self.args.device)
        else:
            selected_goal_index = selected_goal.unsqueeze(-1).repeat(1,self.n_agents).view(-1)
            
        selected_goal_latent = th.gather(goal_latent_exp, 1, selected_goal_index.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.latent_dim)).squeeze(1)
                
        # action selection conditioned on selected goal
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states, selected_goal_latent)
        
        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1) # return qvalues