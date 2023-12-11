import torch.nn as nn


class Prioritizer(nn.Module):
    def __init__(self, encoder, head):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, state):
        state = self.encoder(state)
        value = self.head(state.readout)
        return value
