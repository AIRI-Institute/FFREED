from copy import deepcopy
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from ffreed.utils import lmap


def MolFromSmiles(smile):
    try:
        mol = Chem.MolFromSmiles(smile)
        Chem.SanitizeMol(mol)
    except:
        raise ValueError(f'Failed to sanitize molecule {smile}')
    return mol


def one_hot(x, values, enc2last=False):
    if x not in values:
        if not enc2last:
            raise ValueError(f'{x} not in {values}')
        else:
            x = values[-1]
    return lmap(lambda v: x == v, values)


def brics_compatible(bond_type):
    connection_rules = {
        0: {1, 3, 8},
        1: {0, 2, 11, 12, 13, 14},
        2: {1, 3, 9},
        3: {0, 2, 10, 11, 12, 13, 14},
        4: {11, 12, 13, 14},
        5: {5},
        6: {7, 8, 11, 12, 13, 14},
        7: {6, 11, 12, 13, 14},
        8: {0, 6, 11, 12, 13, 14},
        9: {2, 11, 12, 13, 14},
        10: {3},
        11: {1, 3, 4, 6, 7, 8, 9, 12, 13, 14},
        12: {1, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14},
        13: {1, 3, 4, 6, 7, 8, 9, 11, 12, 14},
        14: {1, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14}
    }
    return connection_rules[bond_type]


def remove_attachments(smile):
    mol = Chem.MolFromSmiles(smile)
    mol = Chem.ReplaceSubstructs(mol, Chem.MolFromSmiles("*"), Chem.MolFromSmiles('[H]'), replaceAll=True)[0]
    mol = Chem.RemoveHs(mol)
    Chem.SanitizeMol(mol)
    smile = Chem.MolToSmiles(mol)
    return smile

def connect_mols(mol1, mol2, atom1, atom2):
    combined = deepcopy(Chem.CombineMols(mol1, mol2))
    emol = Chem.EditableMol(combined)
    neighbor1_idx = atom1.GetNeighbors()[0].GetIdx()
    neighbor2_idx = atom2.GetNeighbors()[0].GetIdx()
    atom1_idx = atom1.GetIdx()
    atom2_idx = atom2.GetIdx()
    bond_order = atom2.GetBonds()[0].GetBondType()
    emol.AddBond(neighbor1_idx, neighbor2_idx +
                    mol1.GetNumAtoms(), order=bond_order)
    emol.RemoveAtom(atom2_idx + mol1.GetNumAtoms())
    emol.RemoveAtom(atom1_idx)
    mol = emol.GetMol()
    return mol


def ecfp(smile, r=2, n=1024):
    smile = remove_attachments(smile)
    molecule = Chem.MolFromSmiles(smile)
    return list(AllChem.GetMorganFingerprintAsBitVect(molecule, r, n))


BRICS_NUM_TYPES = 15
BRICS_MATRIX = np.stack([np.array([j in brics_compatible(i) for j in range(BRICS_NUM_TYPES)]) for i in range(BRICS_NUM_TYPES)])
