"""One allocation from authored prompts, historical metadata and sealed cost evidence."""
import hashlib
import json
import math
import random
import re
from difflib import SequenceMatcher
from itertools import combinations
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SEED = 2026112101
SUMMARY = ROOT.parent / 'paper_results/attribution-20261119/evidence/lawa/summary.json'


def normal(text):
    return ' '.join(re.findall(r'[a-z0-9]+', text.casefold()))


def fields(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield key, value
            yield from fields(value)
    elif isinstance(obj, list):
        for item in obj:
            yield from fields(item)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def history():
    probe = ROOT/'probe_selection/freeze.json'
    manifests = json.loads(probe.read_text())['prior_prompt_manifests'] + ['probe_selection/freeze.json']
    prompts, seeds, records = set(), set(), {}
    for rel in manifests:
        path = ROOT/rel
        data = json.loads(path.read_text())
        pairs = list(fields(data))
        texts = {normal(v) for k, v in pairs if k == 'prompt' and isinstance(v,str)}
        noises = {v for k,v in pairs if k in ('seed','noise_seed','generation_seed','master_seed') and type(v) is int}
        records[rel] = dict(sha256=digest(path), prompts=len(texts), seeds=len(noises),
                            source_image_ids=len(data.get('images', [])))
        prompts.update(texts)
        seeds.update(noises)
    return prompts, seeds, records


def codebooks(rng):
    residues = {x*x % 47 for x in range(1,47)}
    core = [[1 if (j-i)%47 in residues else -1 for j in range(47)] for i in range(47)]
    rows = [[1]+r for r in rng.sample(core,32)]
    layouts = [list(range(48))]+[rng.sample(range(48),48) for _ in range(3)]
    books = {f'H{i}':[''.join('1' if row[j]==1 else '0' for j in order) for row in rows]
             for i,order in enumerate(layouts)}
    books['random'] = [''.join(rng.sample('0'*24+'1'*24,48)) for _ in range(32)]
    # Symmetric truncated arcsine prior, no post-selection of row balance.
    low = math.asin(math.sqrt(1/48))
    biases = [math.sin(rng.uniform(low, math.pi/2-low))**2 for _ in range(48)]
    books['tardos'] = [''.join('1' if rng.random()<p else '0' for p in biases) for _ in range(32)]
    if any(len(set(bank))!=32 for bank in books.values()) or len({frozenset(books[f'H{i}'][:16]) for i in range(4)})!=4:
        raise ValueError('Degenerate draw. Stop.')
    return books, layouts, biases


def budget():
    s = json.loads(SUMMARY.read_text())
    r = s['work_resources']
    totals = dict(generate=8, official_write=432, official_extract=1248, mix=144,
                  quality=432, storage=2496)
    times = {k:r[k]['wall']/r[k]['completed']*v for k,v in totals.items()}
    fixed = s['stages']['model']['wall_seconds']-sum(item['wall'] for item in r.values())
    model = sum(times.values())+max(0,fixed)
    prep = s['stages']['preparation']['wall_seconds']
    return dict(source=str(SUMMARY.relative_to(ROOT.parent)),sha256=digest(SUMMARY),
                historical_resources=s['cumulative_resources'],historical_debits=s['cumulative_budget_debits'],
                projected_components_seconds=times, historical_unallocated_model_seconds=fixed,
                estimated_model_seconds=model, observed_cold_preparation_seconds=prep,
                proposed_wall_caps_seconds=dict(preparation=60*math.ceil(2*prep/60),
                       model=60*math.ceil(2*model/60),settlement=60),
                caps_are_allocated_wall_time=True, GPU_authorized=False)


def build():
    rng = random.Random(SEED)
    texts,seeds,records = history()
    scopes = sorted(texts)
    scenes = (HERE/'scenes.txt').read_text().splitlines()
    assert len(set(map(normal,scenes)))==len(scenes)==48 and not texts.intersection(map(normal,scenes))
    books,layouts,biases = codebooks(rng)
    triples = list(combinations(range(16),3))
    rng.shuffle(triples)
    foreigners = rng.sample(list(combinations(range(16,32),3)),48)
    works = []
    for i,scene in enumerate(rng.sample(scenes,48)):
        split = 'DEV' if i<8 else 'CAL' if i<32 else 'TEST'
        a = triples.pop()
        j = next(j for j,b in enumerate(triples) if not set(a).intersection(b))
        b = triples.pop(j)
        seed = rng.randrange(2**32)
        if seed in seeds: raise ValueError('Seed collision. Stop.')
        seeds.add(seed)
        sim,prior = max((SequenceMatcher(None,normal(scene),t).ratio(),t) for t in scopes)
        works.append(dict(id=f'{split}-{i:02}',split=split,prompt=scene,seed=seed,
                          groups=[a,b,foreigners[i]],book_order=rng.sample(list(books),6),
                          nearest_prior_prompt=prior,character_similarity=round(sim,4)))
    return dict(master_seed=SEED,authored_prompts=digest(HERE/'scenes.txt'),history=records,
                unique_historical_prompts=len(texts),checkpoint_history='UNVERIFIED',
                works=works,books=books,layouts=layouts,tardos_biases=biases,budget=budget())


if __name__=='__main__':
    target=HERE/'allocation.json'
    if target.exists(): raise FileExistsError(target)
    data=build()
    with target.open('x') as f:
        json.dump(data,f,indent=2)
        f.write('\n')
