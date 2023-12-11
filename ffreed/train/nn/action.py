from functools import partial
from operator import attrgetter, methodcaller
import torch.nn.functional as F
import torch

from ffreed.utils import lmap, lzip


class ActionBatch(object):
    def __init__(self, batch_size, index, onehot, embedding, logits):
        self.batch_size = batch_size
        self.index = index.flatten().tolist()
        self.onehot = onehot
        self.embedding = embedding
        self.logits = logits

    def entropy(self):
        bs = self.batch_size
        log_probs = F.log_softmax(self.logits.view(bs, 1, -1), dim=2)
        log_probs = log_probs.masked_fill(log_probs.isinf(), 0)
        entropy = - (log_probs @ self.onehot.view(bs, -1, 1)).squeeze(2)
        return entropy


class StepActionBatch(object):
    def __init__(self, actions):
        assert len(set(map(attrgetter('batch_size'), actions))) == 1
        self.batch_size = actions[0].batch_size
        self.actions = actions
        self._embedding = torch.cat(lmap(attrgetter('embedding'), self.actions), dim=-1)
        self._index = lzip(*map(attrgetter('index'), self.actions))

    def entropy(self):
        return sum(map(methodcaller('entropy'), self.actions))

    @property
    def embedding(self):
        return self._embedding

    @property
    def index(self):
        return self._index
