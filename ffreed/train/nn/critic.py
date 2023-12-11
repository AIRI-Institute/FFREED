from functools import reduce
from operator import methodcaller
import torch
import torch.nn as nn
from torchvision.ops import MLP
import dgl
import numpy as np

from ffreed.utils import lmap, lzip
from ffreed.train.utils import construct_batch, get_attachments


class Critic(nn.Module):
    def __init__(self, encoder, fragments, emb_size=64, n_nets=2, mlp_args=None, mlp_kwargs=None):
        super().__init__()
        self.encoder = encoder
        self.fragments = fragments
        self.emb_size = d = emb_size
        N = len(fragments)

        self.fragments_gcn = [torch.zeros(d) for _ in range(N)]
        self.fragments_attachments = dict()
        self.sections = np.array([len(frag.get_attachments()) for frag in fragments])

        self.n_nets = n_nets
        self.nets = nn.ModuleList([MLP(*mlp_args, **mlp_kwargs) for _ in range(n_nets)])

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.fragments_gcn = [fragment.to(*args, **kwargs) for fragment in self.fragments_gcn]
        return self

    def get_molecule_attachment(self, molecule, index):
        attachments = get_attachments(molecule)
        sections = molecule.sections
        return torch.stack([attachments[i] for i, attachments in zip(index, attachments.split(sections.tolist()))])

    def get_fragment(self, index):
        self.encode_fragments(index)
        return torch.stack([self.fragments_gcn[i] for i in index])

    def get_fragment_attachment(self, fragment_index, attachment_index):
        attachments = torch.cat([self.fragments_attachments[i] for i in fragment_index])
        sections = self.sections[fragment_index]
        return torch.stack([attachments[i] for i, attachments in zip(attachment_index, attachments.split(sections.tolist()))])

    def encode_fragments(self, index):
        index = list(set(index).difference(self.fragments_attachments))
        if not index:
            return
        device = self.fragments_gcn[0].device
        batch = construct_batch([self.fragments[i] for i in index], device=device)
        fragments = self.encoder(batch)
        for i, fragment in zip(index, fragments.readout):
            self.fragments_gcn[i] = fragment
        sections = self.sections[index].tolist()
        for i, attachments in zip(index, get_attachments(fragments).split(sections)):
            self.fragments_attachments[i] = attachments

    def forward(self, state, action, from_index=False):
        state = self.encoder(state)
        if from_index:
            action = lzip(*action)
            ac1, ac2, ac3 = lmap(list, action)
            molecule_attachment = self.get_molecule_attachment(state, ac1)
            fragment = self.get_fragment(ac2)
            fragment_attachment = self.get_fragment_attachment(ac2, ac3)
            action = torch.cat([molecule_attachment, fragment, fragment_attachment], dim=1)
        input = torch.cat([state.readout, action], dim=1)
        values = lmap(methodcaller('forward', input), self.nets)
        return values, reduce(torch.minimum, values, torch.tensor(float("+inf")).to(input.device))

    def reset(self):
        self.fragments_attachments = dict()
        d, N, device = self.emb_size, len(self.fragments), self.fragments_gcn[0].device
        self.fragments_gcn = [torch.zeros(d, device=device) for _ in range(N)]
