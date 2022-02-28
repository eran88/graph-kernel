import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers import GraphConvolution, GraphAggregation


class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)


class Generator(nn.Module):
    """Generator network."""
    def __init__(self, conv_dims, z_dim, vertexes, edges, nodes, dropout):
        super(Generator, self).__init__()

        self.vertexes = vertexes
        self.edges = edges
        self.nodes = nodes

        layers = []
        for c0, c1 in zip([z_dim]+conv_dims[:-1], conv_dims):
            layers.append(nn.Linear(c0, c1))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(p=dropout, inplace=True))
        self.layers = nn.Sequential(*layers)

        self.edges_layer = nn.Linear(conv_dims[-1], edges * vertexes * vertexes)
        self.nodes_layer = nn.Linear(conv_dims[-1], vertexes * nodes)
        self.dropoout = nn.Dropout(p=dropout)

    def forward(self, x):
        output = self.layers(x)
        edges_logits = self.edges_layer(output)\
                       .view(-1,self.edges,self.vertexes,self.vertexes)
        edges_logits = (edges_logits + edges_logits.permute(0,1,3,2))/2
        edges_logits = self.dropoout(edges_logits.permute(0,2,3,1))

        nodes_logits = self.nodes_layer(output)
        nodes_logits = self.dropoout(nodes_logits.view(-1,self.vertexes,self.nodes))

        return edges_logits, nodes_logits


class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self, conv_dim, m_dim, b_dim, dropout,device=False,random_graph_increase=0):
        super(Discriminator, self).__init__()

        graph_conv_dim, aux_dim, linear_dim = conv_dim
        # discriminator
        self.gcn_layer = GraphConvolution(m_dim+random_graph_increase, graph_conv_dim, b_dim, dropout)
        self.agg_layer = GraphAggregation(graph_conv_dim[-1], aux_dim, b_dim+random_graph_increase, dropout)

        # multi dense layer
        layers = []
        for c0, c1 in zip([aux_dim]+linear_dim[:-1], linear_dim):
            layers.append(nn.Linear(c0,c1))
            layers.append(nn.Dropout(dropout))

        self.linear_layer = nn.Sequential(*layers)

        self.output_layer = nn.Linear(linear_dim[-1], 1)
        self.device=device
        self.random_graph_increase=random_graph_increase
    def forward(self, adj, hidden, node, activatation=None):
        adj = adj[:,:,:,1:].permute(0,3,1,2)
        if self.random_graph_increase>0:
            rand=torch.rand(node.shape[0],node.shape[1], self.random_graph_increase,device=self.device)
            node=torch.cat((node,rand),-1)
        annotations = torch.cat((hidden, node), -1) if hidden is not None else node
        h = self.gcn_layer(annotations, adj)
        annotations = torch.cat((h, hidden, node) if hidden is not None\
                                 else (h, node), -1)
        h = self.agg_layer(annotations, torch.tanh)
        h = self.linear_layer(h)

        output = self.output_layer(h)
        output = activatation(output) if activatation is not None else output
        return output, h




class ResBlock(nn.Module):
    def __init__(self, size):
        super(ResBlock, self).__init__()
        self.fc1=nn.Linear(size,size)
        self.fc2=nn.Linear(size,size)
        self.relu=nn.ReLU(inplace=True)
        self.norm1=torch.nn.BatchNorm1d(size)
        self.norm2=torch.nn.BatchNorm1d(size)

    def forward(self, x):
        x=self.relu(self.norm1(self.fc1(x)))
        x=self.norm2(self.fc2(x))
        return x
class Learned_alpha_model(nn.Module):
    def __init__(self, alpha):
        super(Learned_alpha_model, self).__init__()
        real_alpha=1/(2*alpha)
        self.w=nn.Parameter(torch.log(torch.Tensor([real_alpha])))
    def get_alpha(self):
        a = self.w.exp()
        return 1/(a*2)
    def forward(self, x):
        #keep the answer positive
        ans= x*self.w.exp()
        return ans
class Learned_laplacian_model(nn.Module):
    def __init__(self,input_size):
        super(Learned_laplacian_model, self).__init__()
        self.fc0 = nn.Linear(input_size, 100)
        self.fc1 = nn.Linear(100, 100)
        self.fc2 = nn.Linear(100, 10)
        self.bn1 = nn.BatchNorm1d(100)
        self.bn2 = nn.BatchNorm1d(100)
    def forward(self, x):
        x = F.relu(self.bn1(self.fc0(x)))
        x = F.relu(self.bn2(self.fc1(x)))
        x = self.fc2(x)
        return x