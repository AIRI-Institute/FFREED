import argparse
import os
import json
from rdkit import Chem

from ffreed.utils import lmap


def str2strs(s):
    return s.split(',')


def str2floats(s):
    return lmap(float, s.split(','))

    
def str2ints(s):
    return lmap(int, s.split(','))


def str2bool(s):
    if s == 'True':
        return True
    elif s == 'False':
        return False
    else:
        return bool(s)


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--exp_root', type=str, default='/mnt/2tb/experiments/freed')
    parser.add_argument('--commands', type=str2strs, default='train,sample')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--seed', help='RNG seed', type=int, default=666)

    # Environment
    parser.add_argument('--fragments', required=True)
    parser.add_argument('--fragmentation', type=str, default='crem', choices=['crem', 'brics'])
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--starting_smile', type=str, default='c1([*:1])c([*:2])ccc([*:3])c1')
    parser.add_argument('--timelimit', type=int, default=4)

    # model updating
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--actor_lr', type=float, default=1e-4)
    parser.add_argument('--critic_lr', type=float, default=1e-4)
    parser.add_argument('--alpha_lr', type=float, default=5e-4)
    parser.add_argument('--prioritizer_lr', type=float, default=1e-4)
    parser.add_argument('--alpha_eps', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--update_num', type=int, default=256)

    # Saving and Loading
    parser.add_argument('--save_freq', type=int, default=20)
    parser.add_argument('--checkpoint', type=str, default='')

    # SAC
    parser.add_argument('--target_entropy', type=float, default=3.0)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--replay_size', type=int, default=int(1e6))
    parser.add_argument('--polyak', type=float, default=0.995)
    parser.add_argument('--tau', type=float, default=1e-1)
    parser.add_argument('--steps_per_epoch', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--train_alpha', type=str2bool, default=True)

    # Objectives and Rewards
    parser.add_argument('--objectives', type=str2strs, default=['DockingScore'])
    parser.add_argument('--weights', type=str2floats, default=[1.0])
    parser.add_argument('--reward_version', default='hard', choices=['soft', 'hard'])
    parser.add_argument('--alert_collections', type=str)

    # Sample
    parser.add_argument('--num_mols', type=int, default=1000)

    # Docking
    parser.add_argument('--receptor', required=True)
    parser.add_argument('--box_center', required=True, type=str2floats)
    parser.add_argument('--box_size', required=True, type=str2floats)
    parser.add_argument('--vina_program', required=True)
    parser.add_argument('--exhaustiveness', type=int, default=8)
    parser.add_argument('--num_modes', type=int, default=10)
    parser.add_argument('--num_sub_proc', type=int, default=1)
    parser.add_argument('--n_conf', type=int, default=3)
    parser.add_argument('--error_val', type=float, default=99.9)
    parser.add_argument('--timeout_gen3d', type=int, default=None)
    parser.add_argument('--timeout_dock', type=int, default=None)

    # Metrics
    parser.add_argument('--unique_k', type=str2ints, default=[1000, 5000])
    parser.add_argument('--n_jobs', type=int, default=1)

    # Actor and Critic Architectures
    parser.add_argument('--n_nets', type=int, default=2)
    parser.add_argument('--merger', type=str, default='ai', choices=['mi', 'ai'])
    parser.add_argument('--action_mechanism', type=str, default='pi', choices=['sfps', 'pi'])
    parser.add_argument('--ecfp_size', type=int, default=1024)
    parser.add_argument('--emb_size', type=int, default=128)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--aggregation', type=str, default='sum', choices=['sum', 'mean'])

    # PER-PE
    parser.add_argument('--per', type=str2bool, default=False)
    parser.add_argument('--dzeta', type=float, default=0.6)
    parser.add_argument('--beta_start', type=float, default=0.4)
    parser.add_argument('--beta_frames', type=float, default=100000)

    return parser.parse_args()


def update_args(args):
    args.exp_dir = os.path.join(args.exp_root, args.name)

    args.docking_config = get_docking_config(args)

    args.mols_dir = os.path.join(args.exp_dir, 'mols')
    args.model_dir = os.path.join(args.exp_dir, 'ckpt')
    args.logs_dir = os.path.join(args.exp_dir, 'logs')
    args.metrics_dir = os.path.join(args.exp_dir, 'metrics')

    args.atom_vocab = get_atom_vocab()
    args.bond_vocab = get_bond_vocab()
    with open(os.path.join(args.fragments)) as f:
        args.frag_vocab = json.load(f)


def get_docking_config(args):
    docking_config = {
        'receptor': args.receptor,
        'box_center': args.box_center,
        'box_size': args.box_size,
        'vina_program': args.vina_program,
        'exhaustiveness': args.exhaustiveness,
        'num_sub_proc': args.num_sub_proc,
        'num_modes': args.num_modes,
        'timeout_gen3d': args.timeout_gen3d,
        'timeout_dock': args.timeout_dock,
        'seed': args.seed,
        'n_conf': args.n_conf,
        'error_val': args.error_val
    }

    return docking_config


def get_atom_vocab():
    atom_vocab = ['H', 'C', 'N', 'O', 'S', 'P', 'F', 'I', 'Cl', 'Br']
    return atom_vocab


def get_bond_vocab():
    bond_vocab = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ]
    return bond_vocab
