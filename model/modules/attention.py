from torch import nn




class Attention_og(nn.Module):
    """
    A simplified version of attention from DSTFormer that also considers x tensor to be (B, T, J, C) instead of
    (B * T, J, C)
    """

    def __init__(self, dim_in, dim_out, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 mode='spatial'):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_in // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_in, dim_out)
        self.mode = mode
        self.qkv = nn.Linear(dim_in, dim_in * 3, bias=qkv_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, T, J, C = x.shape

        qkv = self.qkv(x).reshape(B, T, J, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2,
                                                                                           5)  # (3, B, H, T, J, C)
        if self.mode == 'temporal':
            q, k, v = qkv[0], qkv[1], qkv[2]
            x = self.forward_temporal(q, k, v)
        elif self.mode == 'spatial':
            q, k, v = qkv[0], qkv[1], qkv[2]
            x = self.forward_spatial(q, k, v)
        else:
            raise NotImplementedError(self.mode)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def forward_spatial(self, q, k, v):
        B, H, T, J, C = q.shape
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, T, J, J)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v  # (B, H, T, J, C)
        x = x.permute(0, 2, 3, 1, 4).reshape(B, T, J, C * self.num_heads)
        return x  # (B, T, J, C)

    def forward_temporal(self, q, k, v):
        B, H, T, J, C = q.shape
        qt = q.transpose(2, 3)  # (B, H, J, T, C)
        kt = k.transpose(2, 3)  # (B, H, J, T, C)
        vt = v.transpose(2, 3)  # (B, H, J, T, C)

        attn = (qt @ kt.transpose(-2, -1)) * self.scale  # (B, H, J, T, T)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ vt  # (B, H, J, T, C)
        x = x.permute(0, 3, 2, 1, 4).reshape(B, T, J, C * self.num_heads)
        return x  # (B, T, J, C)



import torch
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
    def __init__(self, adj, input_dim, output_dim, p_dropout=None):
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


class TPA(nn.Module):
    def __init__(self, adj_temporal, input_dim, output_dim, p_dropout=None):
        super(TPA, self).__init__()

        self.gconv =  LearnableGraphConv(input_dim, output_dim, adj_temporal)
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
    

class Attention(nn.Module):
    def __init__(self, adj, adj_temporal, dim_in, dim_out, 
                 num_heads=8, qkv_bias=False, qk_scale=None, 
                 attn_drop=0., proj_drop=0., mode='spatial'):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_in // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_in, dim_out)
        self.mode = mode
        self.qkv = nn.Linear(dim_in, dim_in * 3, bias=qkv_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        # only create what we need based on mode
        if self.mode == 'spatial':
            self.kpa = KPA(adj=adj, 
                          input_dim=dim_in, 
                          output_dim=dim_in,
                          p_dropout=None)
        elif self.mode == 'temporal':
            self.tpa = TPA(adj_temporal=adj_temporal,
                          input_dim=dim_in,
                          output_dim=dim_in,
                          p_dropout=None)

    def forward(self, x):
        B, T, J, C = x.shape

        if self.mode == 'spatial':
            # KPA needs [B*T, J, C]
            x_in = x.reshape(B*T, J, C)
            x_enriched = self.kpa(x_in)
            # back to [B, T, J, C]
            x_enriched = x_enriched.reshape(B, T, J, C)

        elif self.mode == 'temporal':
            # TPA needs [B*J, T, C]
            x_in = x.permute(0, 2, 1, 3)  # [B, J, T, C]
            x_in = x_in.reshape(B*J, T, C)
            x_enriched = self.tpa(x_in)
            # back to [B, T, J, C]
            x_enriched = x_enriched.reshape(B, J, T, C)
            x_enriched = x_enriched.permute(0, 2, 1, 3)

        # residual addition — safe for first experiment
        x = x + x_enriched

        # rest is exactly same as original
        qkv = self.qkv(x).reshape(
            B, T, J, 3, self.num_heads, C // self.num_heads
        ).permute(3, 0, 4, 1, 2, 5)

        if self.mode == 'temporal':
            q, k, v = qkv[0], qkv[1], qkv[2]
            x = self.forward_temporal(q, k, v)
        elif self.mode == 'spatial':
            q, k, v = qkv[0], qkv[1], qkv[2]
            x = self.forward_spatial(q, k, v)
        else:
            raise NotImplementedError(self.mode)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def forward_spatial(self, q, k, v):
        B, H, T, J, C = q.shape
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        
        x = x.permute(0, 2, 3, 1, 4).reshape(B, T, J, C * self.num_heads)
        return x

    def forward_temporal(self, q, k, v):
        B, H, T, J, C = q.shape
        qt = q.transpose(2, 3)
        kt = k.transpose(2, 3)
        vt = v.transpose(2, 3)
        attn = (qt @ kt.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ vt
        x = x.permute(0, 3, 2, 1, 4).reshape(B, T, J, C * self.num_heads)
        return x
    



# TODO Graph utils functions adding directly here for now, can move to separate file later if needed


import torch
import numpy as np
import scipy.sparse as sp


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def adj_mx_from_edges(num_pts, edges, sparse=True):
    edges = np.array(edges, dtype=np.int32)
    data, i, j = np.ones(edges.shape[0]), edges[:, 0], edges[:, 1]
    adj_mx = sp.coo_matrix((data, (i, j)), shape=(num_pts, num_pts), dtype=np.float32)

    # build symmetric adjacency matrix
    adj_mx = adj_mx + adj_mx.T.multiply(adj_mx.T > adj_mx) - adj_mx.multiply(adj_mx.T > adj_mx)
    adj_mx = normalize(adj_mx) #+ sp.eye(adj_mx.shape[0]))
    if sparse:
        adj_mx = sparse_mx_to_torch_sparse_tensor(adj_mx)
    else:
        adj_mx = torch.tensor(adj_mx.todense(), dtype=torch.float)
    
    adj_mx = adj_mx * (1-torch.eye(adj_mx.shape[0])) + torch.eye(adj_mx.shape[0])

    return adj_mx


def adj_mx_from_skeleton(skeleton):
    num_joints = skeleton.num_joints()
    edges = list(filter(lambda x: x[1] >= 0, zip(list(range(0, num_joints)), skeleton.parents())))
    return adj_mx_from_edges(num_joints, edges, sparse=False)


def adj_mx_from_skeleton_3dhp(num_joints, parents):
    edges = list(filter(lambda x: x[1] >= 0, zip(list(range(0, num_joints)), parents)))
    return adj_mx_from_edges(num_joints, edges, sparse=False)


def adj_mx_from_skeleton_temporal(num_frame, parents):
    num_joints = num_frame
    edges = list(filter(lambda x: x[1] >= 0, zip(list(range(0, num_joints)), parents)))
    return adj_mx_from_edges(num_joints, edges, sparse=False)



def adj_mx_from_skeleton_temporal_extra3(num_frame, parents, parents_extra):
    num_joints = num_frame
    edges = list(filter(lambda x: x[1] >= 0, zip(list(range(0, num_joints)), parents)))
    edges_extra3 = list(filter(lambda x: x[1] >= 0, zip(parents_extra, list(range(0, num_joints-2)))))
    edges_all = np.concatenate((edges, edges_extra3),axis=0)
    return adj_mx_from_edges(num_joints, edges_all, sparse=False)




temporal_skeleton = list(range(0, 27))

temporal_skeleton = np.array(temporal_skeleton)

temporal_skeleton -= 1

#temporal topology

adj_temporal = adj_mx_from_skeleton_temporal(27, temporal_skeleton)


#spatial topology
class Skeleton:
    def __init__(self, parents, joints_left, joints_right):
        assert len(joints_left) == len(joints_right)
        
        self._parents = np.array(parents)
        self._joints_left = joints_left
        self._joints_right = joints_right
        self._compute_metadata()
    
    def num_joints(self):
        return len(self._parents)
    
    def parents(self):
        return self._parents
    
    def has_children(self):
        return self._has_children
    
    def children(self):
        return self._children
    
    def remove_joints(self, joints_to_remove):
        """
        Remove the joints specified in 'joints_to_remove'.
        """
        valid_joints = []
        for joint in range(len(self._parents)):
            if joint not in joints_to_remove:
                valid_joints.append(joint)

        for i in range(len(self._parents)):
            while self._parents[i] in joints_to_remove:
                self._parents[i] = self._parents[self._parents[i]]
                
        index_offsets = np.zeros(len(self._parents), dtype=int)
        new_parents = []
        for i, parent in enumerate(self._parents):
            if i not in joints_to_remove:
                new_parents.append(parent - index_offsets[parent])
            else:
                index_offsets[i:] += 1
        self._parents = np.array(new_parents)
        
        
        if self._joints_left is not None:
            new_joints_left = []
            for joint in self._joints_left:
                if joint in valid_joints:
                    new_joints_left.append(joint - index_offsets[joint])
            self._joints_left = new_joints_left
        if self._joints_right is not None:
            new_joints_right = []
            for joint in self._joints_right:
                if joint in valid_joints:
                    new_joints_right.append(joint - index_offsets[joint])
            self._joints_right = new_joints_right

        self._compute_metadata()
        
        return valid_joints
    
    def joints_left(self):
        return self._joints_left
    
    def joints_right(self):
        return self._joints_right
        
    def _compute_metadata(self):
        self._has_children = np.zeros(len(self._parents)).astype(bool)
        for i, parent in enumerate(self._parents):
            if parent != -1:
                self._has_children[parent] = True

        self._children = []
        for i, parent in enumerate(self._parents):
            self._children.append([])
        for i, parent in enumerate(self._parents):
            if parent != -1:
                self._children[parent].append(i)




h36m_skeleton = Skeleton(parents=[-1,  0,  1,  2,  3,  4,  0,  6,  7,  8,  9,  0, 11, 12, 13, 14, 12,
       16, 17, 18, 19, 20, 19, 22, 12, 24, 25, 26, 27, 28, 27, 30],
       joints_left=[6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22, 23],
       joints_right=[1, 2, 3, 4, 5, 24, 25, 26, 27, 28, 29, 30, 31])


adj = adj_mx_from_skeleton(h36m_skeleton)
