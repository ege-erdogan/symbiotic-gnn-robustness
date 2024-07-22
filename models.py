from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from greatx.nn.models import APPNP
from torch import Tensor
from torch.nn import Linear, Parameter
from torch_geometric.nn import APPNP, GATConv, GCNConv, MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import softmax
from greatx.nn.models import RobustGCN

import torch.nn as nn
from greatx.nn.layers import Sequential, activations

from util import *


def get_model(key=None, dataset=None):
    keys = ["gcn", "gat", "appnp", "gprgnn", "rgcn"]
    assert key in keys or key is None, f"Invalid model key {key}, must be one of {keys}"
    if key == "gcn":
        return GCN(dataset.num_features, 16, dataset.num_classes)
    elif key == "gat":
        return GAT(dataset.num_features, 16, dataset.num_classes)
    elif key == "appnp":
        return APPNPNet(dataset)
    elif key == "gprgnn":
        return GPRGNN(dataset, None)
    elif key == "rgcn":
        return RobustGCN(dataset.num_features, dataset.num_classes)
    return keys


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.norm = gcn_norm
        self.conv1 = GCNConv(in_channels, hidden_channels, normalize=False)
        self.conv2 = GCNConv(hidden_channels, out_channels, normalize=False)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None, penultimate=False, **kwargs):
        # Normalize edge indices only once:
        if not kwargs.get("skip_norm", False):
            edge_index, edge_weight = self.norm(
                edge_index,
                edge_weight,
                num_nodes=x.size(0),
                add_self_loops=True,
            )

        x = self.conv1(x, edge_index, edge_weight).relu()
        if penultimate:
            return x
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x


class WeightedGATConv(GATConv):
    """Extended GAT to allow for weighted edges."""

    def edge_update(
        self,
        alpha_j: Tensor,
        alpha_i: Optional[Tensor],
        edge_attr: Optional[Tensor],
        index: Tensor,
        ptr: Optional[Tensor],
        size_i: Optional[int],
    ) -> Tensor:
        # Given edge-level attention coefficients for source and target nodes,
        # we simply need to sum them up to "emulate" concatenation:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        alpha = F.leaky_relu(alpha, self.negative_slope)

        if edge_attr is not None:
            assert edge_attr.dim() == 1, "Only scalar edge weights supported"
            edge_attr = edge_attr.view(-1, 1)
            # `alpha` unchanged if edge_attr == 1 and -Inf if edge_attr == 0;
            # We choose log to counteract underflow in subsequent exp/softmax
            alpha = alpha + torch.log2(edge_attr)

        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha


class GAT(torch.nn.Module):
    def __init__(
        self, in_channels, hidden_channels, out_channels, heads=8, dropout_p=0.6
    ):
        super().__init__()
        # Initialize edge weights of self-loops with 1:
        self.conv1 = WeightedGATConv(
            in_channels, hidden_channels, fill_value=1.0, heads=heads
        )
        self.conv2 = WeightedGATConv(
            hidden_channels * heads, out_channels, fill_value=1.0, heads=heads
        )
        self.dropout_p = dropout_p

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x


class GPR_prop(MessagePassing):
    """
    propagation class for GPR_GNN
    """

    def __init__(self, K, alpha, Init, Gamma=None, bias=True, **kwargs):
        super(GPR_prop, self).__init__(aggr="add", **kwargs)
        self.K = K
        self.Init = Init
        self.alpha = alpha
        self.Gamma = Gamma

        assert Init in ["SGC", "PPR", "NPPR", "Random", "WS"]
        if Init == "SGC":
            # SGC-like, note that in this case, alpha has to be a integer. It means where the peak at when initializing GPR weights.
            TEMP = 0.0 * np.ones(K + 1)
            TEMP[alpha] = 1.0
        elif Init == "PPR":
            # PPR-like
            TEMP = alpha * (1 - alpha) ** np.arange(K + 1)
            TEMP[-1] = (1 - alpha) ** K
        elif Init == "NPPR":
            # Negative PPR
            TEMP = (alpha) ** np.arange(K + 1)
            TEMP = TEMP / np.sum(np.abs(TEMP))
        elif Init == "Random":
            # Random
            bound = np.sqrt(3 / (K + 1))
            TEMP = np.random.uniform(-bound, bound, K + 1)
            TEMP = TEMP / np.sum(np.abs(TEMP))
        elif Init == "WS":
            # Specify Gamma
            TEMP = Gamma

        self.temp = Parameter(torch.tensor(TEMP))

    def reset_parameters(self):
        torch.nn.init.zeros_(self.temp)
        if self.Init == "SGC":
            self.temp.data[self.alpha] = 1.0
        elif self.Init == "PPR":
            for k in range(self.K + 1):
                self.temp.data[k] = self.alpha * (1 - self.alpha) ** k
            self.temp.data[-1] = (1 - self.alpha) ** self.K
        elif self.Init == "NPPR":
            for k in range(self.K + 1):
                self.temp.data[k] = self.alpha**k
            self.temp.data = self.temp.data / torch.sum(torch.abs(self.temp.data))
        elif self.Init == "Random":
            bound = np.sqrt(3 / (self.K + 1))
            torch.nn.init.uniform_(self.temp, -bound, bound)
            self.temp.data = self.temp.data / torch.sum(torch.abs(self.temp.data))
        elif self.Init == "WS":
            self.temp.data = self.Gamma

    def forward(self, x, edge_index, edge_weight=None):
        edge_index, norm = gcn_norm(
            edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype
        )

        hidden = x * (self.temp[0])
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            gamma = self.temp[k + 1]
            hidden = hidden + gamma * x
        return hidden

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return "{}(K={}, temp={})".format(self.__class__.__name__, self.K, self.temp)


class GPRGNN(torch.nn.Module):
    def __init__(self, dataset, args):
        super(GPRGNN, self).__init__()
        self.lin1 = Linear(dataset.num_features, 16)
        self.lin2 = Linear(16, dataset.num_classes)

        # if args.ppnp == 'PPNP':
        #     self.prop1 = APPNP(10, 0.1)
        # elif args.ppnp == 'GPR_prop':
        self.prop1 = GPR_prop(10, 0.1, "PPR", None)

        self.Init = "PPR"
        self.dprate = 0.5
        self.dropout = 0.5

    def reset_parameters(self):
        self.prop1.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        if self.dprate == 0.0:
            x = self.prop1(x, edge_index, edge_weight)
            return F.log_softmax(x, dim=1)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index, edge_weight)
            return F.log_softmax(x, dim=1)


class APPNPNet(torch.nn.Module):
    def __init__(self, dataset):
        super().__init__()
        self.lin1 = Linear(dataset.num_features, 32)
        self.lin2 = Linear(32, dataset.num_classes)
        self.prop1 = APPNP(10, 0.1)

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, edge_index, edge_weight):
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        x = self.prop1(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)
