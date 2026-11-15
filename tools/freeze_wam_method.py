"""One-time draw using official image metadata; no outcome-dependent selection."""
import hashlib
import json
import sys
import zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from wam_method.protocol import build,rng

def freeze(metadata,history):
    target=ROOT/'wam_method/manifest.json'
    if target.exists():raise RuntimeError('Already frozen; do not redraw')
    metadata=Path(metadata)
    with zipfile.ZipFile(metadata) as z:
        info=json.loads(z.read('annotations/image_info_test2017.json'))
    eligible=sorted([x for x in info['images'] if x['license']==4 and min(x['width'],x['height'])>=256],key=lambda x:x['id'])
    rng('source/coco-test2017').shuffle(eligible)
    images=[dict(id=x['id'],file_name=x['file_name'],width=x['width'],height=x['height'],
        source_url=x['coco_url'],url='https://s3.amazonaws.com/images.cocodataset.org/test2017/'+x['file_name'],
        license=info['licenses'][3]) for x in eligible[:1024]]
    h=json.loads(Path(history).read_text())
    inherited={k:h[k] for k in ['cumulative_resources','cumulative_budget_debits']}
    inherited['history_path']='wam-qualification-20261112-attempt02/summary.json'
    m=build(images,inherited)
    with target.open('x') as f:json.dump(m,f,separators=(',',':'));f.write('\n')
    usage=dict(status='FROZEN_BEFORE_MODEL_OUTCOMES',source='Official COCO image_info_test2017',
        metadata_url='http://images.cocodataset.org/annotations/image_info_test2017.zip',
        transport='https://s3.amazonaws.com/images.cocodataset.org/annotations/image_info_test2017.zip',
        metadata_sha256=hashlib.sha256(metadata.read_bytes()).hexdigest(),
        licenses=info['licenses'],eligibility='license id4 CC BY2.0, minimum dimension256; select by frozen random order',
        eligible_count=len(eligible),selected_count=1024,history='No COCO test2017 use found in available Model-WM QIM/AquaLoRA/WAM manifests; historical source records retained',
        old_dev_works=16,old_aqualora_trajectories=44,old_qim_sources='CIFAR100',
        project_split_disjoint=True,actual_checkpoint_history='UNVERIFIED',
        unseen_content='USER_ACCEPTED_ASSUMPTION',
        limitations='ID-disjoint is not proof against pretrained overlap or all near-duplicates. Exact decoded duplicates stop before training; none are replaced.',
        source_identity='official COCO ID and supplied license metadata; hashes are audit records, not an extra admission gate',
        experiment_seed_unchanged=2026110902,new_domain='Model-WM/writer-abc/v1')
    (ROOT/'wam_method/source_usage_history.json').write_text(json.dumps(usage,indent=2)+'\n')
    print(dict(frozen=str(target),images=len(images),eligible=len(eligible),bytes=target.stat().st_size))

if __name__=='__main__':freeze(*sys.argv[1:])
