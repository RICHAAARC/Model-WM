"""One sampling only. Existing frozen artifacts cause a read-only validation."""
import json
import math
from pathlib import Path
import numpy as np
from scipy.linalg import hadamard

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'latent_control'

def freeze():
    if (DEST/'manifest.json').exists() or (DEST/'frozen.npz').exists():
        from latent_control.protocol import read,arrays,validate
        validate(read(),arrays());print('Existing one-time freeze verified; nothing resampled');return
    seed=2026111601;rng=np.random.Generator(np.random.PCG64(seed))
    subjects=['a ceramic teapot','a wooden sailing boat','a red bicycle','a stone bridge','an old lighthouse',
        'a brass telescope','a glass greenhouse','a woven basket of fruit','a weathered bench','a yellow tram',
        'a hillside cottage','a blue fishing boat','a tiled fountain','a violin on a chair','a book on a desk',
        'a windmill','a small observatory','a parked camper van','a row of terracotta pots','a mountain cabin',
        'a copper kettle','a wildflower garden','an arched doorway','a wooden footbridge','a snowy pine tree',
        'a vintage camera','a folding deck chair','a covered market stall','a small stone tower','a textile loom',
        'a clay sculpture','a railway station clock']
    settings=['beside a quiet lake','near a cobbled courtyard','against distant hills','near a sandy coast',
        'beside an ivy-covered wall','among green trees','in a quiet village','near a riverbank',
        'against pale mountains','beside a brick building','near an open meadow','among autumn leaves',
        'at the edge of a town','near a gravel path','beside a stone terrace','against a cloudy horizon']
    lights=['soft morning light','warm late-afternoon light','diffuse overcast light','cool winter daylight',
        'bright spring daylight','gentle sunset light','clear midday light','misty dawn light']
    views=['a wide view','an eye-level view','a close environmental view','a three-quarter view',
        'a slightly elevated view','a balanced central composition','an off-center composition','a low-angle view']
    slots=rng.choice(32*16*8*8,136,replace=False)
    noise=rng.choice(2**32,136,replace=False)
    old_prompts=set()
    for path in [ROOT/'qualification/manifest.json',ROOT/'wam_qualification/manifest.json',ROOT/'wam_method/manifest.json']:
        def visit(x):
            if isinstance(x,dict):
                if isinstance(x.get('prompt'),str):old_prompts.add(x['prompt'])
                for v in x.values():visit(v)
            elif isinstance(x,list):
                for v in x:visit(v)
        visit(json.loads(path.read_text()))
    permutation=rng.permutation(32768);signs=rng.choice([-1,1],32768).astype(np.int8)
    row_signs=rng.choice([-1,1],64);columns=rng.permutation(64)
    codes=(hadamard(64)*row_signs[:,None])[:,columns].T.astype(np.float32)/8
    key=rng.bytes(32).hex()
    pools=dict(DEV=list(range(16,32)),CAL=list(range(32,48)),TEST=list(range(48,64)))
    works=[]
    for index,(slot,ns) in enumerate(zip(slots,noise)):
        si,rem=divmod(int(slot),16*8*8);se,rem=divmod(rem,8*8);li,vi=divmod(rem,8)
        prompt=f'A realistic photograph of {subjects[si]} {settings[se]}, {lights[li]}, {views[vi]}, natural colors, fine detail.'
        if prompt in old_prompts:raise RuntimeError('Old prompt overlap; stop without replacement sampling')
        split='DEV' if index<8 else ('CAL' if index<72 else 'TEST')
        offset=index if split=='DEV' else (index-8 if split=='CAL' else index-72)
        works.append(dict(id=f'{split}/{offset:03d}',split=split,prompt=prompt,noise_seed=int(ns),
            registered=rng.choice(16,2,replace=False).tolist(),foreign=rng.choice(pools[split],2,replace=False).tolist(),
            arm_order='BC' if index%2==0 else 'CB'))
    old=json.loads((ROOT.parent/'audits/wam-writer-abc-readonly-20261116/summary.json').read_text())
    m=dict(schema=1,run_id='latent-control-20261116-attempt01',master_seed=seed,sampling='PCG64 once, without replacement; no runtime sampling',
        key=key,train_updates=0,authorization='2026-11-16 user adopts METHOD_ROUTE and authorizes implementation/publication; GPU started by user',
        base_repo='stable-diffusion-v1-5/stable-diffusion-v1-5',base_revision='451f4fe16113bff5a5d2269ed5ad43b0592e9a14',
        generation=dict(width=512,height=512,steps=25,cfg=7.5,scheduler='DPMSolverMultistepScheduler',precision='float32'),
        rho=3/255,amplitude=(3/255)*math.sqrt(3*512*512)/2,registered=list(range(16)),foreign_pools=pools,
        dimension=64,identity_capacity_bits=6,mix_weights=[.25,.5,.75],jpeg=dict(quality=85,subsampling=2,optimize=False),
        finite_difference=dict(step=.01,absolute_tolerance=.0005,relative_tolerance=.05),
        gates=dict(joint_catch_min=48,net_gain_min=8,paired_p_max=.05,false_events_max=0,native_ssim=.98,
            psnr_noninferiority_db=.1,ssim_noninferiority=.001,cost_ratio_max=1.),
        physical_limits=dict(generate=136,denoise=3400,write_B=544,write_C=544,forward_C=816,
            finite_difference=16,objective_B=17408,forward_B=17408,backward_B=17408,backward_C=17408,extract=5984),
        budgets=dict(preparation_seconds=1800,model_seconds=21600,cpu_seconds=43200,
            cleanup_margin_seconds=60,ram_bytes=32*1024**3,device_bytes=16*1024**3,result_bytes=10*1024**3),
        inherited=dict(history_path='wam-writer-abc-20261115-attempt01/summary.json',
            cumulative_resources=old['cumulative_resources'],cumulative_budget_debits=old['cumulative_budget_debits']),works=works)
    history=dict(status='FROZEN_BEFORE_REAL_EXECUTION',train=0,dev=8,cal=64,test=64,
        prompt_source='New authored Cartesian vocabulary; uniformly sampled combinations without replacement; generator source retained',
        authoring_banks=dict(subjects=subjects,settings=settings,lights=lights,views=views),
        exact_old_prompt_overlap=0,old_prompts_checked=len(old_prompts),
        prior_used=['both QIM rounds','AquaLoRA all failures and 44 completed trajectories',
            'WAM attempt01 and attempt02 including all failed invocations','WAM writer A/B/C TRAIN/CAL/TEST (closed C_NOT_SUPPORTED)'],
        checkpoint_history='UNVERIFIED',unseen_content='USER_ACCEPTED_ASSUMPTION',
        limitation='New prompt/seed identifiers are independent of past work IDs; pretrained semantic overlap is unverified.',
        no_source_image_download=True,no_prior_TEST_selection=True)
    for name,obj in [('manifest.json',m),('source_usage_history.json',history),('historical_floor.json',old)]:
        with (DEST/name).open('x') as f:json.dump(obj,f,ensure_ascii=False,indent=2);f.write('\n')
    with (DEST/'frozen.npz').open('xb') as f:np.savez_compressed(f,permutation=permutation,signs=signs,codes=codes)
    from latent_control.protocol import validate
    validate(m,dict(permutation=permutation,signs=signs,codes=codes))
    print('One freeze created:',m['run_id'],len(works))

if __name__=='__main__':freeze()
