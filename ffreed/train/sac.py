import time
from collections import defaultdict
from functools import partial
from itertools import chain

import numpy as np
import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from rdkit import Chem
import dgl

from ffreed.train.utils import log_time, log_items, log_info, set_requires_grad, construct_batch
from ffreed.utils import int2str, dump2json
from ffreed.env.utils import remove_attachments


class SAC:
    """
    Soft Actor-Critic (SAC)
    """

    def __init__(self, actor, critic, critic_target, log_alpha, prioritizer, 
                actor_optimizer, critic_optimizer, alpha_optimizer, prioritizer_optimizer, 
                env, replay_buffer, writer, 
                epoch=0, steps_per_epoch=4000, epochs=100,
                gamma=0.99, polyak=0.995, batch_size=100,
                update_num=256, save_freq=5000, train_alpha=True, max_norm=5.,
                device='cpu', target_entropy=1.0, 
                model_dir='.', mols_dir='.', beta_start=0.4, beta_frames=100000,
                **kwargs
    ):
        super().__init__()

        self.actor = actor
        self.critic = critic
        self.critic_target = critic_target
        self.log_alpha = log_alpha
        self.prioritizer = prioritizer

        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.alpha_optimizer = alpha_optimizer
        self.prioritizer_optimizer = prioritizer_optimizer

        self.env = env
        self.replay_buffer = replay_buffer
        self.device = device

        self.gamma = gamma
        self.polyak = polyak
        self.max_norm = max_norm
        self.train_alpha = train_alpha
        self.target_entropy = target_entropy

        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.batch_size = batch_size
        self.update_num = update_num

        self.save_freq = save_freq
        self.model_dir = model_dir
        self.mols_dir = mols_dir
        self.writer = writer

        self.action_dims = [env.action_size, len(env.frag_vocab), env.action_size]

        self.epoch = epoch
        self.beta_start = beta_start
        self.beta_frames = beta_frames

        set_requires_grad(self.critic_target.parameters(), False)

    def critic_loss(self, data):
        weight = data['weight'] if data.get('weight') is not None else 1
        action, state = data['action'], data['state']
        next_state = data['next_state']
        reward, done = data['reward'], data['done']
        q_values, _ = self.critic(state, action, from_index=True)

        with torch.no_grad():
            action = self.actor(next_state)
            entropy = action.entropy()
            _, q_target = self.critic_target(next_state, action.index, from_index=True)
            alpha = self.log_alpha.exp().item()
            target = reward + self.gamma * (1 - done) * (q_target + alpha * entropy)

        loss_critic = sum(map(partial(F.mse_loss, target, reduction='none'), q_values))

        return {
            'critic_loss': (weight * loss_critic).mean()
        }

    def actor_loss(self, data):
        weight = data['weight'] if data.get('weight') is not None else 1
        state = data['state']
        action = self.actor(state)
        _, q_value = self.critic(state, action.embedding)
        alpha = self.log_alpha.exp().item()
        entropy = action.entropy()
        loss_policy = - q_value
        loss_entropy = - alpha * entropy
        loss_alpha = self.log_alpha * (entropy - self.target_entropy).detach()
        loss_actor = loss_entropy + loss_policy
        return {
            'actor_loss': (weight * loss_actor).mean(),
            'entropy_loss': (weight * loss_entropy).mean(),
            'policy_loss': (weight * loss_policy).mean(),
            'alpha_loss': (weight * loss_alpha).mean(),
            'Entropy': entropy.mean(), 
            'Alpha': alpha,
            'Q': q_value.mean()
        }

    def prioritizer_loss(self, data):
        done = data['done'].squeeze(1).to(torch.bool).tolist()
        if not done or not sum(done):
            return {}
        reward = data['reward'][done]
        state = dgl.unbatch(data['state'])
        state = [state for state, done in zip(state, done) if done]
        state = dgl.batch(state)
        predicted = self.prioritizer(state)
        loss_prioritizer = F.mse_loss(predicted, reward, reduction='none')
        error_prioritizer = F.l1_loss(predicted, reward, reduction='none')
        return {
            'prioritizer_loss': loss_prioritizer.mean(),
            'prioritizer_error': error_prioritizer.mean()
        }

    def update_prioritizer(self, data):
        prioritizer_items = self.prioritizer_loss(data)
        self.prioritizer_optimizer.zero_grad()
        if prioritizer_items:
            prioritizer_items['prioritizer_loss'].backward()
            clip_grad_norm_(self.prioritizer.parameters(), self.max_norm)
            self.prioritizer_optimizer.step()
        priority = self.compute_priority(data)
        self.replay_buffer.update_buffer(self.replay_buffer.priority, data['ids'], priority)
        return prioritizer_items

    def update_critic(self, data):
        critic_items = self.critic_loss(data)
        self.critic_optimizer.zero_grad()
        critic_items['critic_loss'].backward()
        clip_grad_norm_(self.critic.parameters(), self.max_norm)
        self.critic_optimizer.step()
        self.critic.reset()
        return critic_items

    def update_actor(self, data):
        actor_items = self.actor_loss(data)
        self.actor_optimizer.zero_grad()
        actor_items['actor_loss'].backward()
        clip_grad_norm_(self.actor.parameters(), self.max_norm)
        self.actor_optimizer.step()
        self.actor.reset()
        return actor_items

    def update_alpha(self, actor_items):
        self.alpha_optimizer.zero_grad()
        actor_items['alpha_loss'].backward()
        self.alpha_optimizer.step()

    def _update(self, data):
        prioritizer_items = dict()
        if self.prioritizer:
            prioritizer_items = self.update_prioritizer(data)
        critic_items = self.update_critic(data)
        set_requires_grad(self.critic.parameters(), False)
        actor_items = self.update_actor(data)
        if self.train_alpha:
            self.update_alpha(actor_items)
        set_requires_grad(self.critic.parameters(), True)
        self.polyak_averaging()
        self.critic_target.reset()
        return {**prioritizer_items, **critic_items, **actor_items}

    @torch.no_grad()
    def polyak_averaging(self):
        for p, p_targ in zip(self.critic.parameters(), self.critic_target.parameters()):
            p_targ.data.mul_(self.polyak)
            p_targ.data.add_((1 - self.polyak) * p.data)

    @torch.no_grad()
    def compute_priority(self, data, batched=True):
        device = self.device
        state, action = data['state'], data['action']
        done, reward = data['done'], data['reward']
        if not batched:
            action = [data['action']]
            state = construct_batch([state], device=device)
        predicted = self.prioritizer(state)
        _, q_value = self.critic(state, action, from_index=True)
        priority = (1 - done) * (q_value - predicted).abs() + done * (reward - predicted).abs()
        return priority.squeeze(dim=1).tolist()

    def compute_batch_weight(self, data):
        N = self.replay_buffer.size
        beta = min(1.0, self.beta_start + N * (1.0 - self.beta_start) / self.beta_frames)
        weight = (data['priority'] * N) ** (-beta)
        return weight

    @log_time
    def update(self):
        log_items = defaultdict(list)
        for _ in range(self.update_num):
            batch = self.replay_buffer.sample_batch(device=self.device, batch_size=self.batch_size)
            if self.prioritizer:
                batch['weight'] = self.compute_batch_weight(batch)
            items = self._update(data=batch)
            for name, value in items.items():
                log_items[name].append(value.item() if torch.is_tensor(value) else value)
        return log_items

    @log_time
    @torch.no_grad()
    def sample(self, num_mols=1, dump=False):
        smiles = list()
        for _ in range(num_mols):
            smile, _ = self.assemble_molecule()
            smiles.append(remove_attachments(smile))
        
        if dump:
            suffix = int2str(self.epoch)
            dump2json(smiles, os.path.join(self.mols_dir, f'sample_{suffix}.json'))
        
        return smiles
    
    def assemble_molecule(self):
        state = self.env.reset()
        done = False
        cnt = 0
        while not done:
            action = self.actor(construct_batch([state], device=self.device))
            next_state, reward, terminated, truncated, info = self.env.step(action.index[0])
            done = terminated or truncated
            experience = {
                'state': state, 'next_state': next_state, 'reward': reward, 'terminated': terminated,
                'truncated': truncated, 'done': done, 'action': action.index[0]
            }
            if self.prioritizer:
                priority = self.compute_priority(experience, batched=False)
                experience['priority'] = priority[0]
            self.replay_buffer.store(experience)
            state = next_state
            cnt += 1
        
        return next_state.smile, cnt

    @log_time
    @torch.no_grad()
    def collect_experience(self):
        smiles, steps = list(), 0
        while steps < self.steps_per_epoch:
            smi, n = self.assemble_molecule()
            smiles.append(remove_attachments(smi))
            steps += n

        rewards = self.compute_rewards(smiles)
        self.update_buffer(rewards['Reward'])
        return {'Smiles': smiles, **rewards}
        
    @log_time
    def compute_rewards(self, smiles):
        return self.env.reward_batch(smiles)
    
    def update_buffer(self, reward):
        buf = self.replay_buffer
        ids = buf.get_update_ids(len(reward))
        buf.update_buffer(buf.reward, ids, reward)

    @log_time
    def train_epoch(self):
        rewards_info = self.collect_experience()
        update_info = self.update()
        return rewards_info, update_info
    
    def save_model(self):
        epoch = self.epoch + 1
        suffix = int2str(epoch)
        fname = os.path.join(self.model_dir, f'model_{suffix}.pth')
        state_dict = {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'alpha_optimizer': self.alpha_optimizer.state_dict(),
            'epoch': epoch,
            'log_alpha': self.log_alpha.item()
        }
        if self.prioritizer:
            state_dict['prioritizer'] = self.prioritizer.state_dict()
            state_dict['prioritizer_optimizer'] = self.prioritizer_optimizer.state_dict()

        torch.save(state_dict, fname)
    
    @log_time
    def train(self):
        suffix = int2str(self.epoch)
        path = os.path.join(self.mols_dir, f'train_{suffix}.csv')
        for epoch in range(self.epoch, self.epochs):
            self.epoch = epoch
            rewards_info, update_info = self.train_epoch()
            log_info(path, rewards_info, epoch, additional_info=update_info, writer=self.writer)
            if epoch % self.save_freq == 0:
                self.save_model()
        
        self.save_model()
        self.epoch += 1