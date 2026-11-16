"""Read the single sampled protocol; never sample in the execution entry."""
import json
from pathlib import Path

RUN_ID = 'latent-control-20261116-attempt01'
PACKAGE = Path(__file__).resolve().parent

def read():
    m = json.loads((PACKAGE/'manifest.json').read_text())
    if m['run_id'] != RUN_ID or m['train_updates'] != 0:
        raise RuntimeError('Wrong frozen protocol')
    return m

def arrays():
    import numpy as np
    with np.load(PACKAGE/'frozen.npz', allow_pickle=False) as z:
        return {k:z[k].copy() for k in z.files}

def validate(m, a):
    import numpy as np
    works = m['works']
    if len(works) != 136 or [sum(w['split']==s for w in works) for s in ['DEV','CAL','TEST']] != [8,64,64]:
        raise RuntimeError('Frozen split mismatch')
    for field in ['id','prompt','noise_seed']:
        if len({w[field] for w in works}) != 136:raise RuntimeError('Repeated work '+field)
    if sorted(a['permutation'].tolist()) != list(range(32768)):raise RuntimeError('Readout permutation')
    if a['signs'].shape != (32768,) or not np.isin(a['signs'],[-1,1]).all():raise RuntimeError('Readout signs')
    if a['codes'].shape != (64,64) or not np.allclose(a['codes']@a['codes'].T,np.eye(64),atol=1e-7):
        raise RuntimeError('Code orthogonality')
    pools=m['foreign_pools']
    if sorted(sum(pools.values(),[])) != list(range(16,64)):raise RuntimeError('Foreign split overlap')
    for w in works:
        if len(set(w['registered']))!=2 or not set(w['registered'])<=set(range(16)):raise RuntimeError('Registered pair')
        if len(set(w['foreign']))!=2 or not set(w['foreign'])<=set(pools[w['split']]):raise RuntimeError('Foreign pair')
    return True
