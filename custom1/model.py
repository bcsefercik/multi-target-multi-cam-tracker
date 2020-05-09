import pdb


import torch
import torch.nn as nn
import dgl.function as fn
from dgl.nn.pytorch import edge_softmax, GATConv

import torch.nn.functional as F


class BCS(nn.Module):
    def __init__(self,
        n_nodes = 4,  
        num_layers=3,
        in_dim=128,
        num_hidden=100,
        g_out_dim=128,
        num_heads=8,
        num_out_heads=1,
        activation=F.elu,
        feat_drop=.3,
        attn_drop=.3,
        negative_slope=.1,
        residual=False):

        super(BCS, self).__init__()
        self.n_nodes = n_nodes
        heads = ([num_heads] * num_layers) + [num_out_heads]
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation))
        # output projection
        self.gat_layers.append(GATConv(
            num_hidden * heads[-2], g_out_dim, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None))

        self.fc1 = nn.Sequential(
            nn.Linear(self.n_nodes*g_out_dim + 128, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 1)
        )


    def forward(self, batch, node_features, f):
        g = batch.graph

        # print(g.batch_size, g.batch_num_nodes, node_features.size(), f.size())
        if g.batch_size != node_features.size()[0]:
            
            appendix = torch.zeros(g.batch_size-node_features.size()[0], 4, 128, requires_grad=True)
            node_features = torch.cat([node_features, appendix])

        for l in range(self.num_layers):
            node_features = self.gat_layers[l](g, node_features).flatten(1)
        node_features = self.gat_layers[-1](g, node_features).mean(1)

        # node_features = node_features.view(-1)
        
        # pdb.set_trace()
        result = torch.FloatTensor([])

        for i in range(f.size()[0]):
            aaa = node_features[self.n_nodes*i:self.n_nodes*(i+1), :]
            rr = self.fc1(torch.cat((node_features[self.n_nodes*i:self.n_nodes*(i+1), :].view(-1), f[i, :]), 0))
            
            result = torch.cat([result, rr], dim=0)

            # pdb.set_trace()
        return result