import json
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
RUN_ID = 'terminal-solver-20261119-attempt01'
GAINS = (0.5, 1, 2)
BUDGET = dict(preparation_seconds=900, model_seconds=10800, cleanup_margin_seconds=60,
              cpu_seconds=14400, ram_bytes=32*2**30, device_bytes=24*2**30, result_bytes=4*2**30)

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf8'))

def protocol():
    original = read_json(PACKAGE.parent/'latent_control/manifest.json')
    allocation = read_json(PACKAGE/'allocation.json')
    assert allocation['run_id']==RUN_ID
    works=allocation['works']
    assert [sum(w['split']==s for w in works) for s in ('DEV','CAL','TEST')]==[8,64,64]
    assert all(len({w[k] for w in works})==136 for k in ('id','family','prompt','noise_seed'))
    for w in works:
        for key,pool in [('registered',original['registered']),('foreign',original['foreign_pools'][w['split']])]:
            assert len(set(w[key]))==2 and set(w[key])<=set(pool)
    shared={k:original[k] for k in ('base_repo','base_revision','generation','amplitude','key',
            'registered','foreign_pools','dimension','identity_capacity_bits','mix_weights','jpeg')}
    frame=(PACKAGE/'scenes.txt').read_text().splitlines()
    assert len(frame)==136 and all(frame[int(w['family'].split('-')[1])-1]==w['prompt'] for w in works)
    return {**shared,**allocation,'budgets':dict(BUDGET),'physical_limits':limits(works),'train_updates':0,
            'quality_protocol':'v2_proxy_only','gates':dict(catch_min=48,net_gain_min=8,
            paired_p_max=.05,union_false_max=0,mean_proxy_increase_max=0)}

def limits(works):
    n=len(works)
    bundles=sum(3 if w['split']=='DEV' else 1 for w in works)
    return dict(generate=n, denoise=25*n, initial_decode=n, center_decode=n, backward_center=64*n,
                backward_C=64*n, backward_O=64*bundles, final_C=4*bundles, closure_O=64*bundles,
                write_C=4*bundles,write_O=4*bundles,extract=44*bundles)
