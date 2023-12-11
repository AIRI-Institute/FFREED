import time
from copy import deepcopy
from operator import attrgetter
from functools import wraps
import numpy as np
import pandas as pd
import torch
import dgl
import random
import torch.nn.functional as F

from ffreed.utils import lmap, dmap


def log_time(method):
    @wraps(method)
    def wrapper(sac, *args, **kwargs):
        t0 = time.time()
        res = method(sac, *args, **kwargs)
        t1 = time.time()
        sac.writer.add_scalar(f'time_{method.__name__}', t1 - t0, sac.epoch)
        return res

    return wrapper


def set_requires_grad(params, value):
    assert isinstance(value, bool)
    for p in params:
        p.requires_grad = value


def log_items(writer, items, iteration):
    def get_item(value):
        if torch.is_tensor(value):
            return value.item()
        elif isinstance(value, (np.ndarray, np.generic, list)):
            return np.mean(value)
        elif isinstance(value, float):
            return value
        elif isinstance(value, pd.Series):
            return value.mean()
        else:
            raise ValueError(f"Items have unsupported '{type(value)}'.")

    items = dmap(get_item, items)
    for name, value in items.items():
        writer.add_scalar(name, value, iteration)


def log_info(path, rewards_info, iteration, additional_info=None, writer=None):
    df = pd.DataFrame(rewards_info)
    df['Epoch'] = iteration
    df.to_csv(path, mode='a', index=False)

    if writer:
        writer.add_text('Samples', df.to_string(index=False), iteration)
        writer.add_scalar("Count", len(df), iteration)
        writer.add_scalar("Unique", len(df['Smiles'].unique()), iteration)
        writer.add_scalar("TotalCount", len(pd.read_csv(path)['Smiles'].unique()), iteration)

        df = df.drop(columns=['Smiles'])
        log_items(writer, df.to_dict(orient='list'), iteration)
        if additional_info:
            log_items(writer, additional_info, iteration)


def construct_batch(states, device='cpu'):
    att_num = list()
    for state in states:
        graph, att_ids, att_types = state.graph, state.attachment_ids, state.attachment_types
        n_nodes = graph.number_of_nodes()
        # dirty hack for batching
        if not (att_ids and att_types):
            att_ids, att_types = [0], [0]

        att_onehot = F.one_hot(torch.LongTensor(att_ids), num_classes=n_nodes)
        graph.ndata['attachment_mask'] = att_onehot.sum(0).bool()
        graph.ndata['attachment_type'] = (att_onehot * torch.LongTensor(att_types)[:, None]).sum(0)
        att_num.append(len(att_ids))

    graphs = lmap(attrgetter('graph'), states)
    batch = deepcopy(dgl.batch(graphs)).to(device)
    batch.sections = torch.LongTensor(att_num).to(device)
    batch.smiles = [state.smile for state in states]
    return batch


def get_attachments(graph, types=False):
    if types:
        embeddings = graph.ndata['attachment_type']
    else:
        embeddings = graph.ndata['x']
    attachment_mask = graph.ndata['attachment_mask']
    return embeddings[attachment_mask]
