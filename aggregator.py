import copy
import math
import torch
import scipy.sparse as sp
import pickle as pk
from GraphGenerator import GraphGenerator
from fed_utilis import sd_matrixing, state_decom
from optimiser import FedProx
import numpy as np
import torch.nn.functional as F
from base_module.GAT import *

def parameter_aggregate(args, A, w_server, global_model, agg_dict, client_recoder, balance_martix):
    # update global weights

    if agg_dict['agg_app'] == 'avg' or agg_dict['agg_app'] == "prox" or agg_dict['agg_app'] == "scaf":
        w_server = average_dic(w_server, "cuda")
        w_server = [w_server] * agg_dict['clients']
        personalized_model = copy.deepcopy(w_server)

    elif agg_dict['agg_app'] == "att":
        w_server = att_dic(w_server, global_model[0], "cuda")
        w_server = [w_server] * agg_dict['clients']
        personalized_model = copy.deepcopy(w_server)

    elif agg_dict['agg_app'] == 'prompt_update':
        personalized_model = fedprompt_attn_agg(w_server, client_recoder, balance_martix, agg_dict)

    return personalized_model

def fedprompt_attn_agg(models_dic, client_recoder=None, balance_martix=None, agg_dict=None):
    # Unsupvised Parameters Reconstrcution
    keys = []
    key_shapes = []
    param_metrix = []
    # models_dic is the weight of server, means that happen mistake/ models_dic = w_server

    for model in models_dic:
        param_metrix.append(state_decom(model, "Transformer_prompt_pre").clone().detach())
    param_metrix = torch.stack(param_metrix)

    for key, param in models_dic[0].items():
        keys.append(key)
        key_shapes.append(list(param.data.shape))

    # GraphGenerator-GAT
    A = generate_adj(param_metrix, agg_dict, agg_dict['clients']).detach().cpu().numpy()
    A = normalize_adj(A)
    A = torch.tensor(A)
    aggregated_param = torch.mm(A, param_metrix)

    # GCN
    for i in range(2):
        aggregated_param = torch.mm(A, aggregated_param)

    aggregated_param = 0.8 * aggregated_param + 0.2 * param_metrix

    for i in range(len(models_dic)):
        pointer = 0
        for k in range(len(keys)):
            num_p = 1
            for n in key_shapes[k]:
                num_p *= n
            models_dic[i][keys[k]] = aggregated_param[i][pointer:pointer + num_p].reshape(key_shapes[k])
            pointer += num_p
    
    return models_dic

def average_dic(model_dic, device, dp=0.001):
    w_avg = copy.deepcopy(model_dic[0])

    for k in w_avg.keys():
        # col = k.split('.')[0]
        # if col == "prompt_len":
        for i in range(1, len(model_dic)):
            w_avg[k] += model_dic[i][k]
        w_avg[k] = torch.div(w_avg[k], len(model_dic))
    return w_avg

def att_dic(w_clients, w_server, device, stepsize=1, metric=1, dp=0.001):
    w_next = copy.deepcopy(w_server)
    att, att_mat = {}, {}
    for k in w_server.keys():
        w_next[k] = torch.zeros_like(w_server[k]).cuda()
        att[k] = torch.zeros(len(w_clients)).cuda()
    for k in w_next.keys():
        for i in range(0, len(w_clients)):
            att[k][i] = torch.norm((w_server[k]-w_clients[i][k]).type(torch.float32), metric)
    for k in w_next.keys():
        att[k] = torch.nn.functional.softmax(att[k], dim=0)
    for k in w_next.keys():
        att_weight = torch.zeros_like(w_server[k])
        for i in range(0, len(w_clients)):
            datatype = w_server[k].dtype
            att_weight += torch.mul(w_server[k] - w_clients[i][k], att[k][i].type(datatype))
        w_next[k] = w_server[k].cuda() - torch.mul(att_weight, stepsize) + torch.mul(torch.randn(w_server[k].shape).cuda(), dp)
    return w_next

def generate_adj(param_metrix, agg_dict, subgraph_size):
    dist_metrix = torch.zeros((len(param_metrix), len(param_metrix)))
    for i in range(len(param_metrix)):
        for j in range(len(param_metrix)):
            dist_metrix[i][j] = torch.nn.functional.pairwise_distance(
                param_metrix[i].view(1, -1), param_metrix[j].view(1, -1), p=2).clone().detach()
    dist_metrix = torch.nn.functional.normalize(dist_metrix).to("cuda")

    """GraphGenerator: Conscturct a adjacent matrix A accoding intilized client latent and uoloaded parameters from each client"""
    gc = GraphGenerator(agg_dict['clients'], subgraph_size, agg_dict['clients'],
                          "cuda", 0.7).to("cuda")
    idx = torch.arange(agg_dict['clients']).to("cuda")
    optimizer = torch.optim.SGD(gc.parameters(), lr=0.001)
    parm = torch.nn.Parameter(torch.empty(88, 88), requires_grad=True)
    stdv4 = 1. / math.sqrt(parm.shape[1])
    parm.data.uniform_(-stdv4, stdv4)
 
    for e in range(10):
        optimizer.zero_grad()
        adj = gc(idx)
        adj = torch.nn.functional.normalize(adj)

        loss = torch.nn.functional.mse_loss(adj, dist_metrix)
        loss.backward()
        optimizer.step()

    adj = gc.eval(idx).to("cpu")


    """GAT: Learning correlation among adjacent client to update adjacent matrix A"""

    adj = adj.detach().numpy()
    adj_coo = sparse.coo_matrix(adj, shape=adj.shape)


    g = dgl.from_scipy(adj_coo)
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    gtn = GAT(g=g, num_layers=2, in_dim=adj_coo.shape[-1], num_hidden=adj_coo.shape[-1], heads=[4, 4], activation=torch.nn.LeakyReLU(),
                feat_drop=0.1, attn_drop=0.1, negative_slope=0.1)
    
    adj = torch.from_numpy(adj)
    dist_metrix = dist_metrix.to("cpu")
    optimizer2 = torch.optim.Adam(gtn.parameters(), lr=0.001)
    for k in range(20):
        k = np.random.randint(low=0, high=2)
        optimizer2.zero_grad()
        head = gtn(adj)
        adj = torch.mul(torch.from_numpy(head[k].detach().numpy()), adj)
        adj = torch.nn.functional.normalize(adj)
        adj = torch.mul(adj, parm)

        loss = torch.nn.functional.mse_loss(adj, dist_metrix)
        loss.requires_grad_(True) 
        loss.backward(retain_graph=True)
        optimizer2.step()
    with torch.no_grad():
        head = gtn(adj)

    adj = torch.sigmoid(torch.from_numpy(head[-1].detach().numpy()) * torch.tanh(adj))
    return adj

def read_out(personalized_models, device):
    # average pooling as read out function
    global_model = average_dic(personalized_models, device, 0)
    return [global_model] * len(personalized_models)

def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx