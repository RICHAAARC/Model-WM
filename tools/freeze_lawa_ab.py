"""Create the LaWa exploratory allocation once, before model use."""
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT/'lawa_ab'/'freeze.json'
SCENES = """An antique dental casting flask sits open on a bed of white investment powder.
A translucent amber traffic delineator folds flat beneath a testing platen.
A chrome harmonica rests in an open black velvet instrument tray.
Two mica sheets peel apart on a geologist's illuminated sample stand.
A darkroom enlarger projects a blank square onto a white easel.
An orange pincushion protea rises from a narrow blue laboratory vase.
A vintage slide rule lies diagonally over a sheet of polar graph paper.
A cream porcelain electrical insulator caps a short weathered crossarm.
An inflatable planetarium dome fills an empty school gym with its entrance tunnel extended.
A red enamel nutcracker clamps a single walnut over a folded linen napkin.
A cylindrical squirrel cage rotor stands upright beside a disassembled electric motor.
A broad rubber squeegee clears one stripe across a fogged shower screen.
A butterfly specimen drawer stands half open under a brass magnifying lens.
An abacus with cobalt beads stands upright at the edge of a slate counter.
A copper lightning conductor rises from the ridge of a green tiled roof.
A scalloped pastry crimper wheel rests beside a freshly sealed semicircular pie.""".splitlines()


def all_values(obj, key):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key:
                yield v
            yield from all_values(v, key)
    elif isinstance(obj, list):
        for v in obj:
            yield from all_values(v, key)


def freeze():
    if TARGET.exists():
        raise FileExistsError('Allocation already frozen')
    seed = 2026111902
    rng = random.Random(seed)
    history = {}
    previous_seeds = set()
    previous_prompts = set()
    for rel in ('qualification/manifest.json','wam_qualification/manifest.json','wam_method/manifest.json',
                'latent_control/manifest.json','terminal_solver/allocation.json'):
        data = json.loads((ROOT/rel).read_text())
        prompts = set(x for x in all_values(data,'prompt') if isinstance(x,str))
        previous_prompts.update(p.casefold().strip() for p in prompts)
        for key in ('seed','noise_seed','generation_seed'):
            previous_seeds.update(x for x in all_values(data,key) if type(x) is int)
        history[rel] = len(prompts)
    assert not previous_prompts.intersection(p.casefold().strip() for p in SCENES)
    scenes = rng.sample(SCENES,len(SCENES))
    codes = [''.join(rng.sample('0'*24+'1'*24,48)) for _ in range(32)]
    assert len(set(codes))==32
    works = []
    for half, group in enumerate(('fit','holdout')):
        for keybase in (0,16):
            pool = list(range(keybase+8*half,keybase+8*half+8))
            rng.shuffle(pool)
            # Each identity occurs twice and each unordered pair occurs once.
            pairs = [[pool[i],pool[(i+1)%8]] for i in range(8)]
            if keybase == 0:
                registered = pairs
            else:
                foreign = pairs
        for i in range(8):
            noise = rng.randrange(2**32)
            assert noise not in previous_seeds
            previous_seeds.add(noise)
            works.append(dict(id=f'{group}-{i:02}',split=group,prompt=scenes[8*half+i],seed=noise,
                              registered=registered[i],foreign=foreign[i]))
    record = dict(seed=seed,codes=codes,registered=list(range(16)),works=works,history_prompt_counts=history,
        sources='Newly authored photographic scenarios; no historical image, caption or latent used.',
        history='Prior QIM CIFAR/COCO, AquaLoRA, WAM, latent-control and terminal-solver including continuation-01 are touched. See terminal_solver/source_history.json. Actual pretraining overlap is unverified.',
        checkpoint_history='UNVERIFIED')
    with TARGET.open('x',encoding='utf8') as out:
        json.dump(record,out,indent=2)
        out.write('\n')


if __name__ == '__main__':
    freeze()
