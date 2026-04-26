import torch
import torch.nn as nn
from mamba_ssm import Mamba


# CONNECTIONS = {10: [9], 9: [8, 10], 8: [7, 9], 
#                14: [15, 8], 15: [16, 14], 
#                11: [12, 8], 12: [13, 11],
#                7: [0, 8], 0: [1, 7], 
#                1: [2, 0], 2: [3, 1], 
#                4: [5, 0], 5: [6, 4], 
#                16: [15], 13: [12], 
#                3: [2], 6: [5]}

CONNECTIONS = {10: [9], 9: [8, 10], 8: [7, 9], 
               14: [15, 8], 15: [16, 14], 
               11: [12, 8], 12: [13, 11],
               7: [0, 8], 0: [1, 7], 
               1: [2, 0], 2: [3, 1], 
               4: [5, 0], 5: [6, 4], 
               16: [15], 13: [12], 
               3: [2], 6: [5]}

# build adj directly from this
def build_adj_from_connections(connections, num_joints=17):
    adj = torch.zeros(num_joints, num_joints)
    for joint, neighbors in connections.items():
        for neighbor in neighbors:
            adj[joint, neighbor] = 1
            adj[neighbor, joint] = 1  # symmetric
    # self connections
    adj = adj + torch.eye(num_joints)
    # normalize rows
    row_sum = adj.sum(dim=1, keepdim=True)
    adj = adj / row_sum
    return adj  # [17, 17]

adj = build_adj_from_connections(CONNECTIONS, num_joints=17)






import torch
import math
class LearnableGraphConv(nn.Module):
    """
    Semantic graph convolution layer
    """

    def __init__(self, in_features, out_features, adj, bias=True):
        super(LearnableGraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(torch.zeros(size=(2, in_features, out_features), dtype=torch.float))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.M = nn.Parameter(torch.ones(size=(adj.size(0), out_features), dtype=torch.float))

        # spatial/temporal local topology
        self.adj = adj
        
        # simulated spatial/temporal global topology
        self.adj2 = nn.Parameter(torch.ones_like(adj))        
        nn.init.constant_(self.adj2, 1e-6)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float))
            stdv = 1. / math.sqrt(self.W.size(2))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        h0 = torch.matmul(input, self.W[0])
        h1 = torch.matmul(input, self.W[1])
        
        adj = self.adj.to(input.device) + self.adj2.to(input.device)
  
        adj = (adj.T + adj)/2
        
        E = torch.eye(adj.size(0), dtype=torch.float).to(input.device)
        
        output = torch.matmul(adj * E, self.M*h0) + torch.matmul(adj * (1 - E), self.M*h1)
        if self.bias is not None:
            return output + self.bias.view(1, 1, -1)
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class KPA(nn.Module):
    def __init__(self, input_dim, output_dim, p_dropout=None, adj = adj):
        super(KPA, self).__init__()

        self.gconv =  LearnableGraphConv(input_dim, output_dim, adj)
        self.bn = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()

        if p_dropout is not None:
            self.dropout = nn.Dropout(p_dropout)
        else:
            self.dropout = None

    def forward(self, x):
        x = self.gconv(x).transpose(1, 2)
        x = self.bn(x).transpose(1, 2)
        if self.dropout is not None:
            x = self.dropout(self.relu(x))

        x = self.relu(x)
        return x



class MambaMixer(nn.Module):
    def __init__(self,
                 dim,
                 mode='spatial',
                 d_state=16,
                 d_conv=4,
                 expand=2,
                 use_kpa=False,
                 use_tpa=False,
                 dropout=0.0):
        super().__init__()


        self.mode = mode
        self.use_kpa = use_kpa
        self.use_tpa = use_tpa

        # optional structure modules
        if use_kpa:
            self.kpa = KPA(input_dim=dim, output_dim=dim, p_dropout=dropout)


        # core mamba
        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )

    def forward(self, x):
        B, T, J, C = x.shape

        if self.mode == 'temporal':
            # mamba over time
            x_in = x.permute(0, 2, 1, 3).reshape(B * J, T, C)
            x_out = self.mamba(x_in)
            x_out = x_out.reshape(B, J, T, C).permute(0, 2, 1, 3)
            return x_out

        elif self.mode == 'spatial':
            # optional spatial structure
            if self.use_kpa:
                x_s = x.reshape(B * T, J, C)
                x_s = self.kpa(x_s)
                x_s = x_s.reshape(B, T, J, C)
                x = x + x_s

            # mamba over joints
            x_in = x.reshape(B * T, J, C)
            x_out = self.mamba(x_in)
            x_out = x_out.reshape(B, T, J, C)
            return x_out

        else:
            raise NotImplementedError(self.mode)