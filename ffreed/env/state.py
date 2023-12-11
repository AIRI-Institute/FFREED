import numpy as np
import dgl
from rdkit import Chem
import torch

from ffreed.utils import lzip
from ffreed.env.utils import one_hot, MolFromSmiles


class State(object):
    def __init__(self, smile, timestamp, fragmentation='crem', atom_dim=None, bond_dim=None, atom_vocab=None, bond_vocab=None, attach_vocab=None):
        self.timestamp = timestamp
        self.molecule = MolFromSmiles(smile)
        self.smile = Chem.MolToSmiles(self.molecule)
        self.fragmentation = fragmentation
        self.atom_dim = atom_dim
        self.bond_dim = bond_dim
        self.atom_vocab = atom_vocab
        self.bond_vocab = bond_vocab
        self.attach_vocab = attach_vocab
        self.attachments = self.get_attachments()
        self.attachment_ids, self.attachment_types = lzip(*self.attachments) if self.attachments else ([], [])
        self.graph = self.mol2graph()
        self.embedding = None

    def atom_feature(self, atom):
        degrees = [0, 1, 2, 3, 4, 5]
        num_hs = [0, 1, 2, 3, 4]
        valencies = [0, 1, 2, 3, 4, 5]
        meta_feature = one_hot(atom.GetDegree(), degrees, enc2last=True) \
                     + one_hot(atom.GetTotalNumHs(), num_hs, enc2last=True) \
                     + one_hot(atom.GetImplicitValence(), valencies, enc2last=True) \
                     + [atom.GetIsAromatic()]

        symbol, smarts = atom.GetSymbol(), atom.GetSmarts()
        type = symbol
        if self.fragmentation == 'brics' and symbol == '*':
            type = smarts
        feature = one_hot(type, self.atom_vocab + self.attach_vocab)

        return feature + meta_feature

    def bond_feature(self, bond):
        return np.asarray(
            one_hot(bond.GetBondType(), self.bond_vocab)
        )

    def mol2graph(self):
        mol = self.molecule
        node_feat = np.empty((mol.GetNumAtoms(), self.atom_dim), dtype=np.float32)
        for a in mol.GetAtoms():
            node_feat[a.GetIdx()] = self.atom_feature(a)

        u, v = list(), list()
        edge_feat = np.empty((mol.GetNumBonds(), self.bond_dim), dtype=np.float32)
        for i, b in enumerate(mol.GetBonds()):
            begin_idx = b.GetBeginAtomIdx()
            end_idx = b.GetEndAtomIdx()
            edge_feat[i] = self.bond_feature(b)
            u.append(begin_idx)
            v.append(end_idx)

        g = dgl.graph((u + v, v + u))
        g.ndata['x'] = torch.from_numpy(node_feat)
        g.edata['x'] = torch.from_numpy(np.concatenate([edge_feat, edge_feat], axis=0))

        return g

    def get_attachments(self):
        attachments = list()
        if self.fragmentation == 'crem':
            att_type = 0
        for atom in self.molecule.GetAtoms():
            if atom.GetSymbol() == '*':
                if self.fragmentation == 'brics':
                    att_type = int(np.argmax(one_hot(atom.GetSmarts(), self.attach_vocab)))
                attachments.append([atom.GetIdx(), att_type])

        return attachments

    def __eq__(self, other):
        return self.smile == other.smile

    def __hash__(self):
        return hash(self.smile)
