import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, use_bias):
        super().__init__()
        
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.use_bias = use_bias
        
        if use_bias:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=True)
        else:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=False)
        
        self.opt_proj = nn.Linear(out_dim * num_heads, in_dim, bias=True)
    
    def forward(self, q, k, v):
        
        q_proj = self.Q(q)
        K_proj = self.K(k)
        V_proj = self.V(v)

        attention = torch.mm(q_proj, K_proj.transpose(0, 1))

        attention = attention - attention.amax(dim = -1, keepdim = True).detach()
        attn = attention.softmax(dim = -1)

        out = torch.mm(attn, V_proj) + q_proj
        out = self.opt_proj(out)
        
        return out
    
class GraphTransformerLayer(nn.Module):
    """
        Param: 
    """
    def __init__(self, in_dim, out_dim, num_heads, param_num, dropout=0.0, layer_norm=False, batch_norm=True, residual=True, use_bias=False):
        super().__init__()

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm        
        self.batch_norm = batch_norm
        self.param_num = param_num
        self.attention = MultiHeadAttentionLayer(in_dim, out_dim//num_heads, num_heads, use_bias)
        

        if self.layer_norm:
            self.layer_norm1 = nn.LayerNorm(out_dim)
            
        if self.batch_norm:
            self.batch_norm1 = nn.BatchNorm1d(out_dim)
        
        # FFN
        self.FFN_layer1 = nn.Linear(out_dim, out_dim*2)
        self.FFN_layer2 = nn.Linear(out_dim*2, out_dim)

        if self.layer_norm:
            self.layer_norm2 = nn.LayerNorm(out_dim)
            
        if self.batch_norm:
            self.batch_norm2 = nn.BatchNorm1d(out_dim)
        
        self.param_proj = nn.Linear(param_num, 207)

    def forward(self, adj_mat, h, param_mat):
        h_in1 = h # for first residual connection
        
        fusion_mat = self.param_proj(torch.mm(adj_mat, param_mat))
        fusion_mat = torch.tanh(fusion_mat)

        # multi-head attention out
        attn_out = self.attention(adj_mat, adj_mat, fusion_mat)
        # 207 207
        h = attn_out.view(-1, self.out_channels)

        h = F.dropout(h, self.dropout, training=self.training)

        
        if self.residual:
            h = h_in1 + h # residual connection
        
        if self.layer_norm:
            h = self.layer_norm1(h)
            
        if self.batch_norm:
            h = self.batch_norm1(h)
        
        h_in2 = h # for second residual connection

        # FFN
        h = self.FFN_layer1(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_layer2(h)

        if self.residual:
            h = h_in2 + h # residual connection
        
        if self.layer_norm:
            h = self.layer_norm2(h)
            
        if self.batch_norm:
            h = self.batch_norm2(h)       

        return h


class graph_transformer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, param_num, dropout=0.0, layer_norm=False, batch_norm=True, residual=True, use_bias=False):
        super(graph_transformer, self).__init__()

        self.layers1 = GraphTransformerLayer(in_dim=in_dim, out_dim=out_dim, num_heads= num_heads, param_num=param_num, layer_norm=True, residual=True)
        self.layers2 = GraphTransformerLayer(in_dim=in_dim, out_dim=out_dim, num_heads= num_heads, param_num=param_num, layer_norm=True, residual=True)
        self.layers3 = GraphTransformerLayer(in_dim=in_dim, out_dim=out_dim, num_heads= num_heads, param_num=param_num, layer_norm=True, residual=True)

    def forward(self, adj_mat, param_mat):
        x = self.layers1(adj_mat, adj_mat, param_mat)
        x2 = self.layers2(x, x, param_mat)
        x3 = self.layers3(x2+x, x2+x, param_mat)

        return F.tanh(x3)

    def eval(self, adj_mat, param_mat):
        x = self.layers1(adj_mat, adj_mat, param_mat)
        x2 = self.layers2(x, x, param_mat)
        x3 = self.layers3(x2+x, x2+x, param_mat)

        return F.tanh(x3)
