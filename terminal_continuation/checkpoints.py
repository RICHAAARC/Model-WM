import math
from zipfile import ZipFile
from pathlib import Path
from terminal_solver.protocol import GAINS, PACKAGE, RUN_ID, limits, read_json

def validate_counts(counts, cap):
    if set(counts)!=set(cap):
        raise ValueError('Missing operation categories')
    for key, maximum in cap.items():
        n=counts[key]
        if type(n) is not int or not 0<=n<=maximum:
            raise ValueError('Invalid operation count: '+key)
        if key not in ('closure_O','backward_O') and n!=maximum:
            raise ValueError('Incomplete work: '+key)
    if counts['closure_O']!=counts['backward_O']:
        raise ValueError('Ordinary closure/gradient mismatch')

def bundle(folder,w,a,g,verify_images=False):
    path=folder/'works'/w['id']/a/str(g)
    b=read_json(path/'readouts.json')
    if b.get('id')!=w['id'] or b.get('arm')!=f'{a}/{g}' or b.get('complete') is not True:
        raise ValueError(f'Missing complete bundle {path}')
    expected={'null':[]}
    for group in ('registered','foreign'):
        for i in range(2):
            expected[f'{group}_{i}']=[w[group][i]] if group=='registered' else []
        for i in range(3):
            expected[f'{group}_mix_{i}']=w[group] if group=='registered' else []
    expected={f'{name}_{codec}':truth for name,truth in expected.items() for codec in ('png','jpeg85')}
    rows=b['rows']
    if len(rows)!=22 or {r['name'] for r in rows}!=set(expected):
        raise ValueError('Incomplete condition family')
    for r in rows:
        if r['truth']!=expected[r['name']] or len(r['scores'])!=16 or not all(map(math.isfinite,r['scores'])):
            raise ValueError('Invalid score or truth row')
        if verify_images:
            filename=r['name']+('.png' if r['name'].endswith('_png') else '.jpg')
            from PIL import Image
            with Image.open(path/filename) as im:
                im.load()
    if not all(math.isfinite(b['proxies'][k]) for k in ('layout','artifact')):
        raise ValueError('Invalid quality measurement')
    return b

def audit_dev(root,m,heartbeat=None):
    root=Path(root)
    old=read_json(root/'summary.json')
    if (old['status']!='STOPPED_PROJECTION' or old['segments']!={'dev':'SELECTED'} or
            old['resources_final'] is not True or old['unsealed'] is not None or (root/'formal').exists()):
        raise ValueError('Expected sealed DEV-only STOPPED_PROJECTION; original run untouched')
    allocation=read_json(root/'allocation.json')
    if allocation['run_id']!=RUN_ID or allocation['works']!=m['works'] or allocation['seed']!=m['seed']:
        raise ValueError('Frozen allocation differs')
    # Compare the deployed reader, solver, physical protocol and identity material
    # with the old notebook's own archived executable sources.
    with ZipFile(root/'executed_source.zip') as deployed:
        for pkg,names in [('latent_control',('frozen.npz','carrier.py','model.py','worker.py','science.py','manifest.json')),
                          ('terminal_solver',('allocation.json','model.py','worker.py','science.py','quality.py','protocol.py'))]:
            for name in names:
                member=pkg+'/'+name
                if deployed.read(member)!=(PACKAGE.parent/member).read_bytes():
                    raise ValueError('Original executed protocol differs: '+member)
    dev=root/'dev'
    works=[w for w in m['works'] if w['split']=='DEV']
    expect=[w['id'] for w in works]
    timing=read_json(dev/'timings.json')
    closed=read_json(dev/'closed.json')
    worker=read_json(dev/'worker_complete.json')
    if ([r['id'] for r in timing]!=expect or worker['works']!=8 or worker['phase']!='dev' or
            closed['status']!='SELECTED'):
        raise ValueError('Incomplete DEV receipts')
    validate_counts(worker['operations'],limits(works))
    op=read_json(dev/'operations_summary.json')
    for k,n in worker['operations'].items():
        if op[k]['completed']!=n or op[k]['started']!=n or op[k]['failed']:
            raise ValueError('Unsettled DEV operation: '+k)
    decision=read_json(dev/'dev_decision.json')
    gains=decision['gains']
    if decision['status']!='SELECTED' or set(gains)!=set('CO') or any(g not in GAINS for g in gains.values()):
        raise ValueError('No sealed DEV gain selection')
    for w in works:
        if heartbeat:
            heartbeat()
        if read_json(dev/'work_receipts'/(w['id'].replace('/','-')+'.json'))['id']!=w['id']:
            raise ValueError('Missing DEV work seal')
        for a in 'CO':
            for g in GAINS:
                bundle(dev,w,a,g,True)
    forecast=read_json(root/'projection.json')
    if forecast['fits'] is not False:
        raise ValueError('Conflicting historical projection and stop')
    return dict(prior_status=old['status'],dev_complete=expect,validated_bundles=48,
                validated_views=1056,gains=gains,original_projection=forecast,
                global_resource_limits=None,scientific_limits=limits([w for w in m['works'] if w['split']!='DEV']),
                change='2026-11-19: report resources; continue CAL/TEST from sealed DEV without rerunning started work.')
