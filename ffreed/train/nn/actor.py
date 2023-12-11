from operator import methodcaller, attrgetter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import MLP
import dgl
from functools import partial
import numpy as np

from ffreed.utils import lmap, lzip
from ffreed.train.nn import Merger, ActionBatch, StepActionBatch
from ffreed.train.utils import construct_batch, get_attachments
from ffreed.env.utils import BRICS_MATRIX, BRICS_NUM_TYPES, ecfp


class Ranker(nn.Module):
    def __init__(self, *, merger_args, merger_kwargs, mlp_args, mlp_kwargs):
        super().__init__()
        self.merger = Merger(*merger_args, **merger_kwargs)
        self.projector = MLP(*mlp_args, **mlp_kwargs)

    def forward(self, x1, x2):
        fused = self.merger(x1, x2)
        logits = self.projector(fused)
        return logits, fused


class Actor(nn.Module):
    def __init__(self, encoder, fragments, emb_size=64, tau=1.0, actions_dim=(40,None,40), 
                 fragmentation='crem', merger='ai', mechanism='pi', ecfp_size=1024, *, mlp_args, mlp_kwargs
    ):
        super().__init__()
        self.encoder  = encoder
        self.fragments = fragments
        self.emb_size = d = emb_size
        self.tau = tau
        self.fragmentation = fragmentation
        self.mechanism = mechanism

        N = len(fragments)
        self.fragments_gcn = [torch.zeros(d) for _ in range(N)]
        self.fragments_attachments = dict()
        attachments = [frag.attachment_types for frag in fragments]
        self.sections = torch.LongTensor(lmap(len, attachments))

        if mechanism == 'pi':
            self.select_fragment = self.select_fragment_PI
            merger_args = ((d, d, d), (d, d, d), (d, d, d))
        elif mechanism == 'sfps':
            self.select_fragment = self.select_fragment_SFPS
            self.fragments_ecfp = torch.FloatTensor(lmap(partial(ecfp, n=ecfp_size), map(attrgetter('smile'), fragments)))
            merger_args = ((d, d, d), (d, ecfp_size, d), (d, d, d))
        else:
            raise ValueError(f"Unknown mechanism '{mechanism}'")

        merger_kwargs = {'fuse': merger}
        self.molecule_attachment_ranker = Ranker(merger_args=merger_args[0], merger_kwargs=merger_kwargs, mlp_args=mlp_args[0], mlp_kwargs=mlp_kwargs[0])
        self.fragment_ranker = Ranker(merger_args=merger_args[1], merger_kwargs=merger_kwargs, mlp_args=mlp_args[1], mlp_kwargs=mlp_kwargs[1])
        self.fragment_attachment_ranker = Ranker(merger_args=merger_args[2], merger_kwargs=merger_kwargs, mlp_args=mlp_args[2], mlp_kwargs=mlp_kwargs[2])

        self.actions_dim = actions_dim

        if self.fragmentation == 'brics':
            self.brics_matrix = torch.from_numpy(BRICS_MATRIX).clone().float()
            self.fragments_attachments_types = [F.one_hot(torch.LongTensor(att), num_classes=BRICS_NUM_TYPES).float() for att in attachments]
            attachments = torch.stack([F.one_hot(torch.LongTensor(list(set(att))), num_classes=BRICS_NUM_TYPES).sum(0) for att in attachments], dim=1).float()
            self.fragments_attachments_compatible = self.brics_matrix[None, :, :] @ attachments[None, :, :]

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.fragments_gcn = [fragment.to(*args, **kwargs) for fragment in self.fragments_gcn]
        if self.mechanism == 'sfps':
            self.fragments_ecfp = self.fragments_ecfp.to(*args, **kwargs)
        self.sections = self.sections.to(*args, **kwargs)
        if self.fragmentation == 'brics':
            self.fragments_attachments_types = [frag.to(*args, **kwargs) for frag in self.fragments_attachments_types]
            self.brics_matrix = self.brics_matrix.to(*args, **kwargs)
            self.fragments_attachments_compatible = self.fragments_attachments_compatible.to(*args, **kwargs)
        return self

    def selected_attachment(self, molecule, index):
        attachments = get_attachments(molecule, types=True).split(molecule.sections.tolist())
        attachment = torch.stack([attachments[i] for attachments, i in zip(attachments, index)])
        attachment = F.one_hot(attachment, num_classes=BRICS_NUM_TYPES).float()
        return attachment

    def acceptable_fragments(self, molecule, index):
        attachment = self.selected_attachment(molecule, index)
        return (attachment[:, None, :] @ self.fragments_attachments_compatible).squeeze(1).gt(0.5)

    def acceptable_sites(self, molecule, attachment_index, fragment_index):
        molecule_attachment = self.selected_attachment(molecule, attachment_index)
        compatible_attachments = molecule_attachment @ self.brics_matrix
        fragment_attachments = [self.fragments_attachments_types[i] for i in fragment_index]
        return torch.cat([att.float() @ comp[:, None] for att, comp in zip(fragment_attachments, compatible_attachments)]).gt(0.5)

    def select_molecule_attachment(self, molecule):
        batch_size = molecule.batch_size
        sections = molecule.sections
        attachments = get_attachments(molecule)
        molecule = molecule.readout.repeat_interleave(sections, dim=0)
        logits, mergers = self.molecule_attachment_ranker(molecule, attachments)
        index, onehot, logits, (attachment, merger) = self.sample_and_pad(self.actions_dim[0], sections.tolist(), logits, attachments, mergers)
        return ActionBatch(batch_size, index, onehot, attachment, logits), merger

    def select_fragment_PI(self, condition, *mask_args, **mask_kwargs):
        batch_size = condition.size(0)
        logits = self.fragment_ranker.projector(condition)
        if self.fragmentation == 'brics':
            mask = ~ self.acceptable_fragments(*mask_args, **mask_kwargs)
            logits.masked_fill_(mask, float("-inf"))
        onehot = F.gumbel_softmax(logits, tau=self.tau, hard=True, dim=1)
        index = torch.argmax(onehot, dim=1)
        self.encode_fragments(index)
        fragment = (onehot[:, None, :] @ torch.stack(self.fragments_gcn)[None, :, :]).squeeze(1)
        merger = self.fragment_ranker.merger(condition, fragment)
        return ActionBatch(batch_size, index, onehot, fragment, logits), merger

    def select_fragment_SFPS(self, condition, *mask_args, **mask_kwargs):
        batch_size, num_frags = condition.size(0), len(self.fragments)
        condition = condition[:, None, :].repeat(1, num_frags, 1)
        fragments = self.fragments_ecfp[None, :, :].repeat(batch_size, 1, 1)
        logits, mergers = self.fragment_ranker(condition, fragments)
        logits = logits.squeeze(2)
        if self.fragmentation == 'brics':
            mask = ~ self.acceptable_fragments(*mask_args, **mask_kwargs)
            logits.masked_fill_(mask, float("-inf"))
        onehot = F.gumbel_softmax(logits, tau=self.tau, hard=True, dim=1)
        index = torch.argmax(onehot, dim=1)
        self.encode_fragments(index)
        fragment = (onehot[:, None, :] @ torch.stack(self.fragments_gcn)[None, :, :]).squeeze(1)
        merger = (onehot[:, None, :] @ mergers).squeeze(1)
        return ActionBatch(batch_size, index, onehot, fragment, logits), merger

    def select_fragment_attachment(self, condition, fragment_index, *mask_args, **mask_kwargs):
        batch_size = condition.size(0)
        attachments = torch.cat([self.fragments_attachments[i] for i in fragment_index])
        sections = self.sections[fragment_index]
        condition = condition.repeat_interleave(sections, dim=0)
        logits, _ = self.fragment_attachment_ranker(condition, attachments)
        if self.fragmentation == 'brics':
            mask = ~ self.acceptable_sites(*mask_args, **mask_kwargs)
            logits.masked_fill_(mask, float("-inf"))
        index, onehot, logits, (attachment, ) = self.sample_and_pad(self.actions_dim[2], sections.tolist(), logits, attachments)
        return ActionBatch(batch_size, index, onehot, attachment, logits)

    def sample_and_pad(self, size, sections, logits, *options):
        batch_size = len(sections)
        options = torch.stack(options, dim=2)
        options = self.pad(options.split(sections), size).view(batch_size, size, -1, options.size(2))
        logits = self.pad(logits.split(sections), size, value=float("-inf")).view(batch_size, size)
        onehot = F.gumbel_softmax(logits, tau=self.tau, hard=True, dim=1)
        index = torch.argmax(onehot, dim=1, keepdim=True)
        options = onehot[None, :, None, :] @ options.permute(3, 0, 1, 2)
        options = [opt for opt in options.squeeze(2)]
        return index, onehot, logits, options

    def pad(self, input, size, value=0):
        def pad(input, size, value=0):
            N, M = input.ndim, input.size(0)
            assert size >= M
            paddings = [0, 0] * (N - 1)
            return F.pad(input, (*paddings, 0, size - M), value=value)
        return torch.cat([pad(x, size, value=value) for x in input], dim=0)

    def encode_fragments(self, index):
        index = index.flatten().tolist()
        index = list(set(index).difference(self.fragments_attachments))
        if not index:
            return
        device = self.fragments_gcn[0].device
        batch = construct_batch([self.fragments[i] for i in index], device=device)
        fragments = self.encoder(batch)
        for i, fragment in zip(index, fragments.readout):
            self.fragments_gcn[i] = fragment
        for i, attachments in zip(index, get_attachments(fragments).split(self.sections[index].tolist())):
            self.fragments_attachments[i] = attachments

    def forward(self, molecule):
        molecule = self.encoder(molecule)
        molecule_attachment, condition = self.select_molecule_attachment(molecule)
        mask_args = (molecule, molecule_attachment.index)
        fragment, condition = self.select_fragment(condition, *mask_args)
        mask_args = (molecule, molecule_attachment.index, fragment.index)
        fragment_attachment = self.select_fragment_attachment(condition, fragment.index, *mask_args)
        return StepActionBatch((molecule_attachment, fragment, fragment_attachment))

    def reset(self):
        self.fragments_attachments = dict()
        d, N, device = self.emb_size, len(self.fragments), self.fragments_gcn[0].device
        self.fragments_gcn = [torch.zeros(d, device=device) for _ in range(N)]
