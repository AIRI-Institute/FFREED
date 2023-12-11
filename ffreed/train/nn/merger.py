import torch.nn as nn


class Merger(nn.Module):
    def __init__(self, n1_in, n2_in, n_out, fuse='mi'):
        super().__init__()
        self.linear1 = nn.Linear(n1_in, n_out)
        self.linear2 = nn.Linear(n2_in, n_out, bias=False)
        self.fuse = fuse
        if fuse == 'mi':
            self.bilinear = nn.Bilinear(n1_in, n2_in, n_out, bias=False)

    def forward(self, x1, x2):
        fused = self.linear1(x1) + self.linear2(x2)
        if self.fuse == 'mi':
            fused = fused + self.bilinear(x1, x2)
        return fused
