import torch.nn as nn
import torch.nn.functional as F
import torch as th

# use goal-embedding for hypernetwork generation

class LAGMAAgent_GC(nn.Module):
    def __init__(self, input_shape, args):
        super(LAGMAAgent_GC, self).__init__()
        self.args = args

        self.goal_repr_Gnet = nn.Sequential(nn.Linear(args.n_cluster, args.goal_repr_dim),
                                        nn.ReLU(),
                                        nn.Linear(args.goal_repr_dim, args.goal_repr_dim))
        input_shape_total = input_shape + args.goal_repr_dim                        
                    
        self.fc1 = nn.Linear(input_shape_total, args.rnn_hidden_dim)            
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)
        
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        
    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()
    
    def forward(self, inputs, hidden_state, goal_one_hot):        
        goal_embed = self.goal_repr_Gnet(goal_one_hot)                
        inputs_total = th.concat((inputs, goal_embed), dim=1).detach()
        
        x = F.relu(self.fc1(inputs_total))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        
        return q, h 