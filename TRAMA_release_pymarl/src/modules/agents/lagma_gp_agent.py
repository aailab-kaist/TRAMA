import torch.nn as nn
import torch.nn.functional as F
import torch as th

class LAGMAAgent_GP(nn.Module):
    def __init__(self, input_shape, args):
        super(LAGMAAgent_GP, self).__init__()
        # goal predictor: same structure with rnn-agent with probability output
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        #self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_codes)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_cluster)
        self.softmax = nn.Softmax(dim=1) # Add softmax layer

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        
        return q, h
    

        
