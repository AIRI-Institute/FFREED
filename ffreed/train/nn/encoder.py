from copy import deepcopy
import torch.nn as nn
from dgl.nn.pytorch.glob import SumPooling

from ffreed.train.nn import GCNLayer


class Encoder(nn.Module):
    def __init__(self, inp_size, emb_size=64, n_layers=3, aggregation='sum'):
        super().__init__()
        self.emb_size = emb_size
        self.emb_linear = nn.Linear(inp_size, emb_size, bias=False)
        self.gcn_layers = nn.ModuleList([GCNLayer(emb_size, emb_size, aggregation=aggregation, residual=False)])
        for _ in range(n_layers - 1):
            self.gcn_layers.append(GCNLayer(emb_size, emb_size, aggregation=aggregation))
        self.pooling = SumPooling()

    def forward(self, graph):
        graph = deepcopy(graph)
        graph.ndata['x'] = self.emb_linear(graph.ndata['x'])
        for conv in self.gcn_layers:
            graph.ndata['x'] = conv(graph)
        graph.readout = self.pooling(graph, graph.ndata['x'])
        return graph
