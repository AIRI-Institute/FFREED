import torch
import torch.nn as nn
import dgl.function as fn


class GCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, aggregation='sum', residual=True):
        super().__init__()
        self.residual = residual
        assert aggregation in ['sum', 'mean'], "Wrong aggregation type"
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.activation = nn.ReLU()
        self.message = fn.copy_src(src='x', out='m')

        def reduce_mean(nodes):
            accum = torch.mean(nodes.mailbox['m'], 1)
            return {'x': accum}

        def reduce_sum(nodes):
            accum = torch.sum(nodes.mailbox['m'], 1)
            return {'x': accum}

        self.aggregation = reduce_mean if aggregation == 'mean' else reduce_sum

    def forward(self, graph):
        identity = graph.ndata['x']
        graph.update_all(self.message, self.aggregation)
        out = self.linear(graph.ndata['x'])
        out = self.activation(out)
        if self.residual:
            out = out + identity
        return out
