from multiprocessing import Pool
import numpy as np
from moses.utils import mapper
from moses.utils import disable_rdkit_log, enable_rdkit_log
from moses.metrics import get_all_metrics, \
                     compute_intermediate_statistics, \
                     fraction_passes_filters, \
                     internal_diversity, \
                     fraction_unique, \
                     fraction_valid, \
                     remove_invalid, \
                     FCDMetric, \
                     SNNMetric, \
                     FragMetric, \
                     ScafMetric
from moses.metrics.utils import compute_scaffolds, get_mol
from moses.metrics import WassersteinMetric, weight, logP, SA, QED


def compute_metrics(gen, ref=None, k=(1000, 5000), batch_size=512, n_jobs=1, add_metrics=None):
    """
    Computes all available metrics between reference sets
    and generated sets of SMILES.
    Parameters:
        gen: list of generated SMILES
        ref: dict of reference lists of SMILES
        k: int or list with values for unique@k. Will calculate number of
            unique molecules in the first k molecules. Default [1000, 10000]
        batch_size: batch size for FCD metric
        add_metrics: dict of additional metrics
    Available metrics:
        * %valid
        * %unique@k
        * Frechet ChemNet Distance (FCD)
        * Fragment similarity (Frag)
        * Scaffold similarity (Scaf)
        * Similarity to nearest neighbour (SNN)
        * Internal diversity (IntDiv)
        * Internal diversity 2: using square root of mean squared
            Tanimoto similarity (IntDiv2)
        * %passes filters (Filters)
        * Distribution difference for logP, SA, QED, weight
    """
    disable_rdkit_log()
    metrics = {}
    pool = Pool(n_jobs)
    metrics['valid'] = fraction_valid(gen, n_jobs=pool)
    gen = remove_invalid(gen, canonize=True)
    for _k in k:
        metrics['unique@{}'.format(_k)] = fraction_unique(gen, _k, pool)
    mols = mapper(pool)(get_mol, gen)

    for name, test in ref.items():
        ptest = compute_intermediate_statistics(test,
                                                batch_size=batch_size,
                                                pool=pool)
        test_mols = mapper(pool)(get_mol, test)
        test_scaffolds = list(compute_scaffolds(test_mols, n_jobs=pool).keys())
        ptest_scaffolds = compute_intermediate_statistics(
            test_scaffolds,
            batch_size=batch_size,
            pool=pool
        )
        metrics[f'FCD/{name}'] = FCDMetric(batch_size=batch_size)(gen=gen, pref=ptest['FCD'])
        metrics[f'SNN/{name}'] = SNNMetric(batch_size=batch_size)(gen=mols, pref=ptest['SNN'])
        metrics[f'Frag/{name}'] = FragMetric(batch_size=batch_size)(gen=mols, pref=ptest['Frag'])
        metrics[f'Scaf/{name}'] = ScafMetric(batch_size=batch_size)(gen=mols, pref=ptest['Scaf'])
        if ptest_scaffolds is not None:
            metrics[f'FCD/{name}SF'] = FCDMetric(batch_size=batch_size)(
                gen=gen, pref=ptest_scaffolds['FCD']
            )
            metrics[f'SNN/{name}SF'] = SNNMetric(batch_size=batch_size)(
                gen=mols, pref=ptest_scaffolds['SNN']
            )
            metrics[f'Frag/{name}SF'] = FragMetric(batch_size=batch_size)(
                gen=mols, pref=ptest_scaffolds['Frag']
            )
            metrics[f'Scaf/{name}SF'] = ScafMetric(batch_size=batch_size)(
                gen=mols, pref=ptest_scaffolds['Scaf']
            )

        # Properties
        for fname, func in [('logP', logP), ('SA', SA),
                        ('QED', QED),
                        ('weight', weight)]:
            metrics[f'{fname}/{name}'] = WassersteinMetric(func, batch_size=batch_size)(
                gen=mols, pref=ptest[fname])
    
    if add_metrics:
        for name, metric in add_metrics.items():
            metrics[name] = np.mean(list(map(metric, mols)))

    metrics['IntDiv'] = internal_diversity(mols, pool)
    metrics['IntDiv2'] = internal_diversity(mols, pool, p=2)
    metrics['Filters'] = fraction_passes_filters(mols, pool)

    enable_rdkit_log()
    pool.close()
    pool.join()
    return metrics
