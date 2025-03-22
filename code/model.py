import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GCN2Conv, GATConv, global_max_pool as gmp, global_add_pool as gap, global_mean_pool as gep, global_sort_pool
from torch_geometric.utils import dropout_adj, softmax
from torch_geometric.nn import GCNConv, GCN2Conv, GATConv, global_max_pool as gmp, global_add_pool as gap, global_mean_pool as gep, global_sort_pool
from torch_geometric.nn import HypergraphConv
from torch_geometric.nn import GraphNorm
from torch_geometric.nn import SAGPooling, GCNConv, SAGEConv, global_mean_pool as gap, global_max_pool as gmp
import argparse
import os.path as osp
from typing import Any, Dict, Optional

import torch
from torch.nn import (
    BatchNorm1d,
    Embedding,
    Linear,
    ModuleList,
    ReLU,
    Sequential,
)
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import GraphNorm
import torch_geometric.transforms as T
from torch_geometric.datasets import ZINC
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_add_pool
import inspect
from typing import Any, Dict, Optional
from torch_geometric.utils import to_dense_batch

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HypergraphConv
from torch_geometric.nn import GCNConv, GCN2Conv, GATConv, global_max_pool as gmp, global_add_pool as gap, global_mean_pool as gep, global_sort_pool
from torch_geometric.utils import dropout_adj, softmax
from torch_geometric.nn import SAGPooling
device = 'cuda:0'
class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=64):
        super(Attention, self).__init__()
        self.project_x = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )
        self.project_xt = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, x, xt):
        x = self.project_x(x)
        xt = self.project_xt(xt)
        a = torch.cat((x, xt), 1)
        a = torch.softmax(a, dim=1)
        return a
    
class pro_encoder(torch.nn.Module):
    def __init__(self, n_output=1, num_features_pro=33, output_dim=128, dropout=0.2):
        super(pro_encoder, self).__init__()
        self.pro_conv = nn.ModuleList([])
        self.pro_out_feats = 128
        self.pro_conv.append(GCNConv(num_features_pro, self.pro_out_feats))
        self.pro_conv.append(GATConv(self.pro_out_feats, self.pro_out_feats, heads=2, dropout=dropout, concat=False))
        self.pro_conv.append(GATConv(self.pro_out_feats, self.pro_out_feats, heads=2, dropout=dropout, concat=False))
        self.pro_seq_fc1 = nn.Linear(self.pro_out_feats, self.pro_out_feats)
        self.pro_seq_fc2 = nn.Linear(self.pro_out_feats, self.pro_out_feats)
        self.pro_bias = nn.Parameter(torch.rand(1, self.pro_out_feats))
        torch.nn.init.uniform_(self.pro_bias, a=-0.2, b=0.2)
        self.relu = nn.ReLU()
    def forward(self, target_x, target_edge_index, target_weight, perturbation=False):
        # 蛋白质图编码
        pro_n = target_x.size(0)
        for i in range(len(self.pro_conv)):
            if i == 0:
                xt = self.pro_conv[i](target_x, target_edge_index, target_weight)
            else:
                xt = self.pro_conv[i](target_x, target_edge_index)
            if i < len(self.pro_conv) - 1:
                xt = self.relu(xt)
            if i == 0:
                target_x = xt
                continue
            pro_z = torch.sigmoid(
                self.pro_seq_fc1(xt) + self.pro_seq_fc2(target_x) + self.pro_bias.expand(pro_n, self.pro_out_feats))
            target_x = pro_z * xt + (1 - pro_z) * target_x
            if perturbation:
                random_noise = torch.rand_like(target_x).to(target_x.device)
                target_x = target_x + torch.sign(target_x) * F.normalize(random_noise, dim=-1) * 0.1
        return target_x
    
class mol_encoder(torch.nn.Module):
    def __init__(self, n_output=1, num_features_mol=78, output_dim=128, dropout=0.2):
        super(mol_encoder, self).__init__()
        self.mol_conv = nn.ModuleList([])
        self.mol_out_feats = 256
        self.mol_conv.append(GATConv(num_features_mol, self.mol_out_feats, heads=2, dropout=dropout, concat=False))
        self.mol_conv.append(GATConv(self.mol_out_feats, self.mol_out_feats, heads=2, dropout=dropout, concat=False))
        self.mol_conv.append(GATConv(self.mol_out_feats, self.mol_out_feats, heads=2, dropout=dropout, concat=False))
        self.mol_seq_fc1 = nn.Linear(self.mol_out_feats, self.mol_out_feats)
        self.mol_seq_fc2 = nn.Linear(self.mol_out_feats, self.mol_out_feats)
        self.mol_bias = nn.Parameter(torch.rand(1, self.mol_out_feats))
        torch.nn.init.uniform_(self.mol_bias, a=-0.2, b=0.2)
        self.Hyper_conv = nn.ModuleList([])
        self.mol_out_feats_hyper = 128
        self.Hyper_conv.append(HypergraphConv(self.mol_out_feats + num_features_mol, self.mol_out_feats_hyper, dropout=0.2))
        self.Hyper_conv.append(HypergraphConv(self.mol_out_feats_hyper, self.mol_out_feats_hyper, dropout=0.2))
        self.relu = nn.ReLU()
    def forward(self, mol_x, mol_edge_index, hyper_edge):
        hyper_x = mol_x.clone().detach()
        hyper_x = hyper_x.to(device)
        # 简单图编码
        mol_n = mol_x.size(0)
        for i in range(len(self.mol_conv)):
            x = self.mol_conv[i](mol_x, mol_edge_index)
            if i < len(self.mol_conv)-1:
                x = self.relu(x)
            if i==0:
                mol_x = x
                continue
            mol_z = torch.sigmoid(self.mol_seq_fc1(x) + self.mol_seq_fc2(mol_x) + self.mol_bias.expand(mol_n, self.mol_out_feats))
            mol_x = mol_z * x + (1 - mol_z) * mol_x
        mol_x_copy = mol_x.clone().detach()
        mol_x_copy = mol_x_copy.to(device)
        hyper_x = torch.cat([mol_x_copy, hyper_x], dim=1)
        for i in range(len(self.Hyper_conv)):
            hyper_x = self.Hyper_conv[i](hyper_x, mol_edge_index)
            hyper_x = self.relu(hyper_x)
        mol_x = torch.cat([mol_x, hyper_x], dim=1)
        return mol_x

class Cross_Attention(torch.nn.Module):
    def __init__(self, embed_dim=128, num_heads=4, kdim=128, vdim=128):
        super(Cross_Attention, self).__init__()
        """ Cross multi-head attention module """
        self.multihead_attn_p_to_d = nn.MultiheadAttention(embed_dim=embed_dim * 3, num_heads=num_heads, kdim=kdim, vdim=vdim)
        self.multihead_attn_d_to_p = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, kdim=embed_dim * 3, vdim=embed_dim * 3)
    def forward(self, compound_feature, prot_feature, vertex_mask, seq_mask):
        vertex_mask = vertex_mask.bool() == False
        seq_mask = seq_mask.bool() == False

        compound_att_output, _ = self.multihead_attn_p_to_d(query=compound_feature.transpose(1,0), key=prot_feature.transpose(1,0),\
                                                      value=prot_feature.transpose(1,0), key_padding_mask=seq_mask.bool())

        prot_att_output, _ = self.multihead_attn_d_to_p(query=prot_feature.transpose(1,0), key=compound_feature.transpose(1,0),\
                                                      value=compound_feature.transpose(1,0), key_padding_mask=vertex_mask.bool())
         
        return compound_att_output.transpose(1,0), prot_att_output.transpose(1,0)

class concat_predictDTA(torch.nn.Module):
    def __init__(self, n_output=1, embed_dim=128, dropout=0.2):
        super(concat_predictDTA, self).__init__()
        self.attention = Attention(embed_dim)
        self.fc1 = nn.Linear(2 * embed_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, n_output)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.mol_fc_g1 = torch.nn.Linear(128 * 3, 1024)
        self.mol_fc_g2 = torch.nn.Linear(1024, embed_dim)
        self.pro_fc_g1 = torch.nn.Linear(128, 1024)
        self.pro_fc_g2 = torch.nn.Linear(1024, embed_dim)
    def forward(self, target_x, target_batch, mol_x, mol_batch):
        mol_x = gmp(mol_x, mol_batch)
        target_x = gmp(target_x, target_batch)
        mol_x = self.relu(self.mol_fc_g1(mol_x))
        mol_x = self.dropout(mol_x)
        mol_x = self.mol_fc_g2(mol_x)
        mol_x = self.dropout(mol_x)
        target_x = self.relu(self.pro_fc_g1(target_x))
        target_x = self.dropout(target_x)
        target_x = self.pro_fc_g2(target_x)
        target_x = self.dropout(target_x)
        att = self.attention(mol_x, target_x)
        emb = torch.stack([mol_x, target_x], dim=1)
        att = att.unsqueeze(dim=2)
        xc = (att * emb).reshape(-1, 2 * 128)
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        
        return out, att

class cross_predictDTA(torch.nn.Module):
    def __init__(self, n_output=1, embed_dim=128, dropout=0.2):
        super(cross_predictDTA, self).__init__()
        
        self.attention = Attention(embed_dim)
        self.cross_attention = Cross_Attention()
        self.fc1 = nn.Linear(2 * embed_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, n_output)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.mol_fc_g1 = torch.nn.Linear(128 * 3, 1024)
        self.mol_fc_g2 = torch.nn.Linear(1024, embed_dim)
        self.pro_fc_g1 = torch.nn.Linear(128, 1024)
        self.pro_fc_g2 = torch.nn.Linear(1024, embed_dim)
        
    def forward(self, target_x, target_batch, mol_x, mol_batch):
        prot_feature, seq_mask = to_dense_batch(target_x, target_batch, max_num_nodes=1000)
        atom_feature, vertex_mask = to_dense_batch(mol_x, mol_batch, max_num_nodes=268)
        
        compound_att_output, prot_att_output = self.cross_attention(atom_feature, prot_feature, vertex_mask, seq_mask)
        
        compound_att_output = torch.sum(compound_att_output, dim=1) / torch.sum(vertex_mask.unsqueeze(dim=-1), dim=1)
        prot_att_output = torch.sum(prot_att_output, dim=1) / torch.sum(seq_mask.unsqueeze(dim=-1), dim=1)
        
        mol_x = compound_att_output + gmp(mol_x, mol_batch)
        target_x = prot_att_output + gmp(target_x, target_batch)
        
        mol_x = self.relu(self.mol_fc_g1(mol_x))
        mol_x = self.dropout(mol_x)
        mol_x = self.mol_fc_g2(mol_x)
        mol_x = self.dropout(mol_x)
        target_x = self.relu(self.pro_fc_g1(target_x))
        target_x = self.dropout(target_x)
        target_x = self.pro_fc_g2(target_x)
        target_x = self.dropout(target_x)
        
        att = self.attention(mol_x, target_x)
        emb = torch.stack([mol_x, target_x], dim=1)
        att = att.unsqueeze(dim=2)
        xc = (att * emb).reshape(-1, 2 * 128)
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        
        return out, att

class Graph_GAT(torch.nn.Module):
    def __init__(self, n_output=1, num_features_pro=33, num_features_mol=78, num_features_xc=92, embed_dim=128, output_dim=128, n_filters=32, dropout=0.2):
                           
        super(Graph_GAT, self).__init__()
        print('DTA_GAT Loading ...')
        self.n_output = n_output
        self.Mol_encoder = mol_encoder()
        self.Pro_encoder = pro_encoder()
        self.Concat_PredictDTA = concat_predictDTA()
        self.Cross_PredictDTA = cross_predictDTA()
                           
    def forward(self, data_mol, data_pro, Cross_bool=False):
        mol_x, mol_edge_index, mol_batch, hyper_edge, target_seq, mol_seq = data_mol.x, data_mol.edge_index, data_mol.batch, data_mol.hyperedge_index, data_mol.target_seq, data_mol.mol_seq
        X_diff_mol1 = data_mol.x_diff_mol1
        X_diff_mol2 = data_mol.x_diff_mol2
        target_x, target_edge_index, target_weight, target_batch = data_pro.x, data_pro.edge_index, data_pro.edge_weight, data_pro.batch
        target_edge_index_svd, target_weight_svd = data_pro.edge_index_svd, data_pro.edge_weight_svd
        
        
        # 特征提取
        mol_x_emb = self.Mol_encoder(mol_x, mol_edge_index, hyper_edge)
        target_x_emb = self.Pro_encoder(target_x, target_edge_index, target_weight)
                           
        # 交叉注意力融合预测
        if Cross_bool:
            out1, att = self.Cross_PredictDTA(target_x_emb, target_batch, mol_x_emb, mol_batch)
            out2, _ = self.Concat_PredictDTA(target_x_emb, target_batch, mol_x_emb, mol_batch)
            out = out1 * 0.5 + out2 * 0.5
            return out, att
        else:
            # 药物加入对比学习
            drug_diff1 = self.Mol_encoder(X_diff_mol1, mol_edge_index, hyper_edge)
            drug_diff1 = gmp(drug_diff1, mol_batch)  # global pooling
            drug_diff2 = self.Mol_encoder(X_diff_mol2, mol_edge_index, hyper_edge)
            drug_diff2 = gmp(drug_diff2, mol_batch)  # global pooling
            # 蛋白质加入对比学习
            xt_svd = self.Pro_encoder(target_x, target_edge_index_svd, target_weight_svd, perturbation=False)
            xt_svd = gmp(xt_svd, target_batch)  # global pooling
            xt_random = self.Pro_encoder(target_x, target_edge_index, target_weight, perturbation=True)
            xt_random = gmp(xt_random, target_batch)  # global pooling
            
            out, att = self.Concat_PredictDTA(target_x_emb, target_batch, mol_x_emb, mol_batch)   
            return out, att, xt_svd, xt_random, drug_diff1, drug_diff2