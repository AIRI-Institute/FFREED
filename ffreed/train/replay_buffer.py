import numpy as np
import torch
import os
import dgl
import json
from functools import reduce, partial
from ffreed.utils import lmap
import pickle
from ffreed.utils import int2str
from ffreed.train.utils import construct_batch


class ReplayBuffer:
    __buffer_names__ = set({'state', 'next_state',  'action', 'reward', 'terminated', 'truncated', 'done'})

    def __init__(self, size, checkpoint=None, priority=False, dzeta=0.6):
        if priority:
            self.__buffer_names__.add('priority')

        for buffer_name in self.__buffer_names__:
            setattr(self, buffer_name, list())

        self.dzeta = dzeta
        self.size, self.max_size = 0, size
        if checkpoint:
            self.load(checkpoint)

    def store(self, experience):
        assert experience.keys() == self.__buffer_names__
        if self.size == self.max_size:
            for buffer_name in __buffer_names__:
                buffer = getattr(self, buffer_name)
                buffer.pop(0)

        for buffer_name in self.__buffer_names__:
            buffer = getattr(self, buffer_name)
            buffer.append(experience[buffer_name])

        self.size = min(self.size + 1, self.max_size)

    def get_update_ids(self, n):
        cnt = 0
        done = self.done
        i = len(done) - 1
        ids = list()
        while cnt < n:
            if done[i]:
                ids.append(i)
                cnt += 1
            i -= 1
        
        ids.reverse()
        return ids

    @staticmethod
    def update_buffer(buffer, ids, values):
        for idx, val in zip(ids, values):
            buffer[idx] = val

    @staticmethod
    def delete_multiple_element(list_object, indices):
        indices = sorted(indices, reverse=True)
        for idx in indices:
            if idx < len(list_object):
                list_object.pop(idx)

    def sample_batch(self, device='cpu', batch_size=32):
        if 'priority' in self.__buffer_names__:
            priority = np.array(self.priority) ** self.dzeta
            priority = priority / priority.sum()
            idxs = np.random.choice(self.size, size=batch_size, p=priority)
        else:
            idxs = np.random.randint(0, self.size, size=batch_size)

        state = construct_batch([self.state[idx] for idx in idxs], device=device)
        next_state = construct_batch([self.next_state[idx] for idx in idxs], device=device)
        action = [self.action[idx] for idx in idxs]

        reward = torch.FloatTensor([self.reward[idx] for idx in idxs])[:, None].to(device)
        terminated = torch.FloatTensor([self.terminated[idx] for idx in idxs])[:, None].to(device)
        truncated = torch.FloatTensor([self.truncated[idx] for idx in idxs])[:, None].to(device)
        done = torch.FloatTensor([self.done[idx] for idx in idxs])[:, None].to(device)

        batch = {
            'ids': idxs,
            'state': state,
            'next_state': next_state,
            'reward': reward,
            'terminated': terminated,
            'truncated': truncated,
            'done': done,
            'action': action
        }
        if 'priority' in self.__buffer_names__:
            batch['priority'] = torch.FloatTensor(priority[idxs])[:, None].to(device)

        return batch

    def save(self, path):
        self._mkdirs(path)
        base = int(math.log(self.max_size), 10)
        for buf_name in ['state', 'next_state']:
            for i, state in enumerate(getattr(self, buf_name)):
                with open(os.path.join(path, buf_name, f'{int2str(i, base)}.pickle'), 'wb') as f:
                    pickle.dump(state, f)
        np.save(os.path.join(path, 'reward.npy'), np.array(self.reward, dtype=np.float32))
        np.save(os.path.join(path, 'terminated.npy'), np.array(self.terminated, dtype=np.bool))
        np.save(os.path.join(path, 'truncated.npy'), np.array(self.truncated, dtype=np.bool))
        np.save(os.path.join(path, 'done.npy'), np.array(self.done, dtype=np.bool))
        torch.save({'action': torch.cat(self.action)}, os.path.join(path, 'action.pth'))

    def _mkdirs(self, path):
        for buf_name in ['state', 'next_state']:
            os.makedirs(os.path.join(path, buf_name), exist_ok=True)

    def _save(self, items, path, name):
        with open(os.path.join(path, f'{name}.json'), 'wt') as f:
            json.dump(items, f)

    def load(self, path):
        self.reward = np.load(os.path.join(path, 'reward.npy')).tolist()
        self.terminated = np.load(os.path.join(path, 'terminated.npy')).tolist()
        self.truncated = np.load(os.path.join(path, 'truncated.npy')).tolist()
        self.done = np.load(os.path.join(path, 'done.npy')).tolist()
        self.size = len(self.rew)
        self.action = torch.load(os.path.join(path, 'action.pth'))['action']
        self.action = list(torch.split(self.action, len(self.action) // self.size))
        base = int(math.log(self.max_size), 10)
        for buf_name in ['state', 'next_state']:
            states = list()
            for i in range(self.size):
                with open(os.path.join(path, buf_name, f'{int2str(i, base)}.pickle'), 'rb') as f:
                    state = pickle.load(f)
                states.append(state)
            setattr(self, buf_name, states)

    def _load(self, path, name):
        with open(os.path.join(path, f'{name}.json'), 'rt') as f:
            items = json.load(f)
        return items
