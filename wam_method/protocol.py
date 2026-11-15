"""Frozen experiment construction. This module never opens model outputs."""
import hashlib
import json
import random
from pathlib import Path

RUN_ID = 'wam-writer-abc-20261115-attempt01'
SEED = 2026110902
SOURCE_COMMIT = '2c08af04d037d5667c02f6ddebbda9ff04581c3e'
CHECKPOINT_URL = 'https://dl.fbaipublicfiles.com/watermark_anything/wam_mit.pth'
CONDITIONS = ['null_png','null_jpeg85','registered_0_png','registered_1_png',
              'foreign_0_png','foreign_1_png','registered_mix_png','registered_mix_jpeg85',
              'foreign_mix_png','foreign_mix_jpeg85']
RISK_CONDITIONS = ['null_png','null_jpeg85','registered_mix_png','registered_mix_jpeg85',
                   'foreign_mix_png','foreign_mix_jpeg85']

def rng(label):
    return random.Random(int.from_bytes(hashlib.sha256(f'Model-WM/writer-abc/v1|{SEED}|{label}'.encode()).digest(),'big'))

def bundle(label, images):
    r = rng(label)
    biases = [min(.95,max(.05,r.betavariate(.7,.7))) for _ in range(128)]
    def book():
        return [''.join(str(int(r.random()<p)) for p in biases) for _ in range(16)]
    return dict(id=label,images=images,biases=biases,registered=book(),foreign=book(),
                registered_pair=sorted(r.sample(range(16),2)),foreign_pair=sorted(r.sample(range(16),2)),
                key=r.getrandbits(256).to_bytes(32,'big').hex())

def messages(b, position, source='registered'):
    return [[int(v) for v in b[source][u][position*32:(position+1)*32]] for u in b[source+'_pair']]

def build(images, inherited):
    assert len(images)==1024 and len({x['id'] for x in images})==1024
    order=list(images);rng('images').shuffle(order)
    splits={'TRAIN':order[:256],'CAL':order[256:512],'TEST':order[512:]}
    train=[]
    for step in range(500):
        ids=[x['id'] for x in rng(f'train/images/{step}').sample(splits['TRAIN'],4)]
        train.append(bundle(f'TRAIN/{step:04d}',ids))
    evaluation={s:[bundle(f'{s}/{i:03d}',[x['id'] for x in splits[s][4*i:4*i+4]])
                   for i in range(len(splits[s])//4)] for s in ['CAL','TEST']}
    return dict(schema=1,run_id=RUN_ID,master_seed=SEED,
        authorization='2026-11-15 user agrees to modified WAM A/B/C proposal; GPU user-started',
        source_commit=SOURCE_COMMIT,checkpoint_url=CHECKPOINT_URL,
        checkpoint_history='UNVERIFIED',unseen_content='USER_ACCEPTED_ASSUMPTION',
        inherited=inherited,images=order,splits={s:[x['id'] for x in v] for s,v in splits.items()},
        train=train,evaluation=evaluation,conditions=CONDITIONS,risk_conditions=RISK_CONDITIONS,
        preprocessing=dict(size=512,rule='Pillow RGB; resize shortest edge to 512 with bicubic; center crop 512'),
        training=dict(updates_per_arm=500,bundle_size=4,learning_rate=5e-5,optimizer='AdamW',
            weight_decay=.01,betas=[.9,.999],eps=1e-8,gradient_clip_norm=1.,
            native_bce=1.,mixed_bce=1.,mask_bce=1.,image_mse=100.,tail_lambda=1.,
            tail_epsilon=1e-6,precision='float32',passes_per_work=2,
            jpeg_rule='even update PNG, odd update actual Pillow JPEG85 forward with identity STE',
            quantization='native save_image round; mixture rint; identity STE inside clamp',
            checkpoint='final update only',hyperparameter_trials=1),
        physical_limits=dict(train_updates_B=500,train_updates_C=500,train_write=16000,
            train_extract=24000,evaluation_write=9216,evaluation_extract=23040,generate=0),
        budgets=dict(gpu_occupancy_seconds=21600,wall_seconds=28800,cpu_seconds=21600,
            preparation_seconds=1800,train_arm_seconds=3600,evaluation_seconds=10800,
            cleanup_margin_seconds=60,result_bytes=20*1024**3,ram_bytes=32*1024**3,
            device_bytes=24*1024**3,download_reporting_only=True),
        gates=dict(native_bit_accuracy=.95,mean_psnr_db=23.,mean_ssim=.75,each_native_psnr_db=20.,
            psnr_noninferiority_db=.5,ssim_noninferiority=.005,min_net_wins=13,
            paired_p=.05,risk_upper_95=.05,max_compute_overhead=.05),
        calibration='strict > maximum of 64 CAL bundle union-innocent maxima; separately A/B/C',
        training_authorized=True,cal_test_authorized=True,vae_authorized=False,
        reuse_attempt=False,retry=False,scientific_pass=None)

def read():
    return json.loads(Path(__file__).with_name('manifest.json').read_text())
