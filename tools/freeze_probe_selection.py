"""One allocation only. Do not run on an existing freeze."""
import json
import random
from pathlib import Path
from freeze_lawa_ab import all_values

ROOT = Path(__file__).resolve().parents[1]
SCENES = """A green bookbinding press holds a stack of marbled endpapers.
A railway semaphore arm stands beside a snow-covered signal cabin.
A handbell with a wooden handle rests inside a felt lined case.
An upright bass bridge lies on the luthier's measuring bench.
An old cream separator stands on the threshold of a stone dairy.
A tidal staff gauge rises beside a bolted iron mooring ring.
A pear-shaped separatory funnel hangs above an empty conical flask.
A black shoe last sits between leather pattern pieces.
A circular stained glass rose window casts colored patches onto tiles.
A bronze ship telegraph stands on the corner of a wheelhouse.
A weaving shuttle loaded with red yarn rests across a linen warp.
A backpacking stove heats a dented kettle on a rocky ledge.
An overhead projector illuminates a transparent leaf specimen on its glass stage.
A canary-yellow luggage trolley carries a striped hatbox on a tiled platform.
An open case contains a metal bore gauge and numbered sleeves.
A notched cheese trier rests on the rind of an aging wheel.
A punch clock holds a single cardboard time card beside a factory doorway.
A geodesic surveying prism reflects sunlight atop a white pole.
A round bread proofing basket holds a spiral of flour on linen.
A red fire hose reel hangs above a painted concrete sill.
A canvas falconry hood rests on a stitched leather glove.
A sheet of handmade paper hangs dripping from a wooden drying frame.
An enamel camping plate holds a single split pomegranate.
A butter churn with a tall dash stands on a flagstone floor.
A mechanical metronome stands beside a folded blue music stand.
A wool comb with long steel tines rests across a fleece-covered table.
A glass hydrometer floats upright in a tall amber sample cylinder.
A hinged oyster basket hangs from a timber pier at low tide.
A cream-colored reel of perforated paper tape feeds through a brass reader.
An inverted wicker lobster pot rests on wet granite.
A small cast iron fly press stands beside a tray of copper blanks.
A bright yellow quoit hangs from the peg of a weathered outdoor board.""".splitlines()

def main():
    dest = ROOT/'probe_selection/freeze.json'
    if dest.exists():
        raise FileExistsError(dest)
    old = json.loads((ROOT/'lawa_ab/freeze.json').read_text())
    prior = list(old['history_prompt_counts']) + ['lawa_ab/freeze.json']
    prompts, seeds = set(), set()
    for path in prior:
        data=json.loads((ROOT/path).read_text())
        prompts.update(v.strip().casefold() for v in all_values(data,'prompt') if isinstance(v,str))
        for key in ('seed','noise_seed','generation_seed'):
            seeds.update(v for v in all_values(data,key) if type(v) is int)
    assert len(set(SCENES)) == 32 and not prompts.intersection(s.casefold() for s in SCENES)
    seed=2026111903
    rng=random.Random(seed)
    prompts_new=rng.sample(SCENES,32)
    probe=[''.join(rng.sample('0'*24+'1'*24,48)) for _ in range(16)]
    assert len(set(probe+old['codes']))==48
    dev=[]
    for row in old['works'][:8]:
        ids=list(row['registered'])
        ids.append(rng.choice(sorted(set(range(8))-set(ids))))
        dev.append(dict(id=row['id'],reg=ids))
    works=[]
    for i,prompt in enumerate(prompts_new):
        split='cal' if i<24 else 'explore'
        noise=rng.randrange(2**32)
        assert noise not in seeds
        seeds.add(noise)
        works.append(dict(id=f'{split}-{i if i<24 else i-24:02}',split=split,prompt=prompt,
                          seed=noise,reg=rng.sample(range(16),3),foreign=rng.sample(range(16,32),3)))
    record=dict(seed=seed,actions=['native','fine','coarse'],transfer_fraction=.10,
                quality_ratio=1.10,probe_codes=probe,codes=old['codes'],probe_triples=[[0,1,2],[3,4,5]],
                prior_prompt_manifests=prior,dev=dev,works=works)
    with dest.open('x') as handle:
        json.dump(record,handle,indent=2);handle.write('\n')

if __name__=='__main__':
    main()
