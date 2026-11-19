"""The historical fit PNGs/latents are read-only inputs. New output has one reservation."""
from pathlib import Path
from lawa_ab import entry
from . import owner
from .worker_paths import PRIOR, RUN_ID

PACKAGE=Path(__file__).resolve().parent

def history(parent):
    old=entry.read(parent/PRIOR/'summary.json')
    if old.get('resources_final') is not True or old.get('status')!='AB_COMPLETE':
        raise ValueError('Historical LaWa receipt is incomplete')
    return dict(source=PRIOR,**{k:old[k] for k in ('cumulative_resources','cumulative_budget_debits')})

def work_resources(out):
    result={}
    for file in sorted((out/'work_receipts').glob('*/usage.json')):
        for kind,row in entry.read(file)['times'].items():
            value=result.setdefault(kind,dict(wall=0.,cpu=0.,completed=0))
            for key in value:
                value[key]+=row[key]
    return result

def run_colab(parent='/content/drive/MyDrive/Model-WM',runtime='/content/model-wm-runtime',**timing):
    base=Path(parent)
    out=base/RUN_ID
    if not out.exists():
        old=base/PRIOR
        required=[old/'summary.json',old/'freeze.json']
        required += [old/'fit'/f'fit-{i:02}'/name for i in range(8)
                     for name in ('latent.npy','null_png.png','reg_0_png.png','reg_1_png.png')]
        missing=[str(p) for p in required if not p.is_file()]
        if missing:
            return dict(status='MISSING_HISTORICAL_INPUT',missing=missing,output=str(out),reserved=False)
        history(base)
        if entry.read(old/'freeze.json')!=entry.read(entry.PACKAGE/'freeze.json'):
            raise ValueError('Historical allocation differs from the published LaWa run')
    return owner.run_colab(parent,runtime,run_id=RUN_ID,package=PACKAGE,history_reader=history,
                           denominator=40,artifacts=('freeze.json','PREREGISTRATION.md'),
                           usage_reader=work_resources,**timing)
