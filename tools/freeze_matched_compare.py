"""Run once before CAL/TEST; audit all historical and reserved allocations."""
import json, hashlib, random, re
from pathlib import Path
from difflib import SequenceMatcher

ROOT=Path(__file__).resolve().parents[1]
TARGET=ROOT/'matched_compare/allocation.json'
PROMPTS='''A brass ship telegraph stands on a wooden bridge beside a rain streaked window.
A translucent insect wing is held between two microscope slides on a black laboratory mat.
A stone watermill wheel rests in a shallow stream beside a moss covered sluice.
A pair of curling stones sits on a freshly swept ice rink beside a red brush.
A glass distillation column rises above a copper heating vessel in a workshop.
A mechanical cash register with round keys sits on a green marble shop counter.
A basket of harvested artichokes stands beside a muddy garden fork in a shed.
A suspended hang glider casts a triangular shadow on the floor of an aircraft hangar.
A brass tuba lies inside an open velvet lined case on a concert stage.
A white satellite dish points above a desert research cabin at dawn.
A broken terracotta roof tile reveals layers of red clay on a masonry bench.
A wooden sled with curved runners leans against a snow covered log cabin.
A stainless steel centrifuge holds six capped tubes in an open circular rotor.
A blue enamel camping kettle hangs over an unlit stone fire ring.
A striped beach windbreak surrounds a low folding chair on an empty sandy shore.
A ceiling mounted surgical light illuminates an empty stainless instrument tray.
A stack of slate roofing samples sits beside a brass thickness gauge.
A snowshoe hare stands near a fallen birch branch in a winter clearing.
A kiln shelf holds small glazed ceramic test cones in an orderly line.
A large chalk compass lies open beside a half drawn circle on a classroom board.
A concrete wave breaker protrudes from calm harbor water beside a narrow ladder.
A reel of perforated cinema film sits on a backlit editing table.
A pale green accordion rests on a stool beside a closed theater curtain.
A hydraulic bottle jack supports an iron beam in a clean maintenance bay.
A shallow tray of amber beads sits beside a jeweler's hand drill.
A portable seismograph records a thin line inside a mountain observation hut.
A cluster of purple cabbages fills a raised garden bed beneath a mesh canopy.
A polished steel propeller blade rests on a padded inspection cradle.
A paper lantern shaped like a fish hangs above an empty courtyard.
A clay amphora is half buried in sand beside a small archaeological brush.
A pair of fencing masks hangs from hooks above a wooden practice floor.
A tall column of stacked basalt stones stands in a misty tidal pool.
A red fire hydrant stands beside a rolled canvas hose in a brick alley.
A tufted loom shuttle lies across a tightly woven blue tapestry.
A spherical glass fishing float rests in a knotted rope net on a dock.
A freestanding harp casts long string shadows across a sunlit wall.
A shallow drawer displays labeled mineral streak plates and a porcelain tile.
A ceramic coffee dripper rests above a clear flask on a stone counter.
A black anemometer spins above a fenced coastal weather station.
A cylindrical bamboo bird cage hangs empty beneath a tiled veranda.
A pair of stained glass panels stands against a white workshop wall.
A wooden bowling pin sits beside a scarred black bowling ball on an old lane.
A hospital wheelchair is parked beside a folded blue privacy screen.
A narrow glass terrarium contains a single fern and layers of pale gravel.
A pair of alpine skis lies across a red roof rack in a snowy parking area.
A mortar and pestle holds coarse green pigment beside a folded linen cloth.
A vertical metal letterpress chase holds tightly packed movable type.
A titanium bicycle crank rests beside a disassembled chain on a repair stand.
A yellow survey buoy floats beside a reed covered lake bank.
A row of small glass hourglasses stands on a dark wooden classroom shelf.
A brass sextant is arranged in an open fitted navigation box.
A coil of thick climbing rope hangs from a carabiner on a rough cliff wall.
A handmade felt hat rests on a wooden milliner's block beside a steam spout.
A clear acrylic ant farm shows winding tunnels filled with reddish sand.
A long wooden canoe rests upside down on two trestles beside a lake.
A white plaster ionic capital sits on a pallet in a sculpture workshop.
A chain operated block hoist hangs from a steel ceiling beam.
A folded paper map sits beside a magnetic compass on a rock ledge.
A cluster of round seed pods spills from a muslin sack onto a potting bench.
A hammered copper gong hangs from a dark timber frame in an empty room.
A laboratory burette releases a drop into a conical flask containing pale pink liquid.
A portable welding helmet stands beside unused leather gloves on a steel table.
A mechanical metronome rests on a closed upright piano in a music room.
A rusty railway signal lever stands in a small brick signal cabin.
A cork bulletin board displays an arrangement of dried seaweed specimens.
A ceramic bird bath catches raindrops beneath a flowering apple tree.
A pair of long wooden oars is mounted above a round porthole.
A shallow pan of indigo dye stands beside an uncolored cotton scarf.
A box of metal watch springs sits under a jeweler's magnifying lens.
A glass prism projects a narrow rainbow onto a white laboratory card.
A row of rubber printing rollers rests on a splattered workshop shelf.
A large terracotta chimney pot stands beside a coiled roofer's rope.
A manual grain mill is clamped to a rough wooden kitchen table.
A bronze sundial stands on a square pedestal in a gravel garden.
A pair of leather falconry gloves rests beside a wooden bird perch.
A compact green winch pulls a steel cable across a repair shop floor.
A row of handmade soap bars dries on a woven reed mat.
A stone spiral staircase curves around a dim central shaft.
A single diving flipper lies beside a mesh equipment bag on a pool deck.
A shallow copper pan contains white beeswax pellets and a wooden stirring stick.
A glass harmonica with nested bowls rests in a wooden display cabinet.
A broad leaf catches a cluster of dew droplets beside a blurred greenhouse window.
A fabric parachute canopy is spread across a grassy training field.
A stainless pastry cutter rests beside a neat row of unbaked triangular pastries.
A small wooden water tower stands above a miniature railway track.
A round tambourine with brass jingles hangs from a theater dressing room hook.
A stone mortar wall contains a small arched opening with a closed iron gate.
A pale blue enamel washbasin sits on a three legged wooden stand.
A copper lightning rod rises from the peak of a weathered barn roof.
A loom reed and two wooden bobbins lie on a clean white workbench.
A red ceramic teapot sits beside a bamboo tea scoop on a slate tray.
A large glass aquarium contains a sunken ceramic arch and fine white gravel.
A folded striped deckchair leans against a pale yellow seaside hut.
A mechanical platform scale holds a sack of dried beans in an open warehouse.
A bundle of oboe reeds rests in a narrow velvet lined storage box.
A stainless laboratory sieve holds a small mound of pale mineral powder.
A silver moka pot sits beside two empty espresso cups on a tiled counter.
A wedge shaped wooden doorstop holds open a heavy green workshop door.
A white paper architectural model shows an empty courtyard surrounded by arches.
A row of ice skates hangs by their laces above a rubber rink mat.
A polished stone grinding wheel is mounted on a cast iron hand crank frame.
A single orange life ring hangs against the blue side of a small ferry.
A red lacquered abacus rests on a folded cotton cloth beside an ink stone.
A mechanical seed drill is parked beside parallel furrows in a dry field.'''.splitlines()

def collect(x,prompts,seeds):
    if isinstance(x,dict):
        for k,v in x.items():
            if k=='prompt' and isinstance(v,str): prompts.add(' '.join(v.lower().split()))
            elif 'seed' in k and isinstance(v,int): seeds.add(v)
            else: collect(v,prompts,seeds)
    elif isinstance(x,list):
        for v in x:collect(v,prompts,seeds)

def main():
    if TARGET.exists():raise FileExistsError('Already frozen; no redraw')
    dev=json.loads((ROOT/'matched_loss/allocation.json').read_text())
    paths=list(dev['history'])+['matched_loss/allocation.json']
    prompts=set();seeds=set();history={}
    for name in paths:
        data=(ROOT/name).read_bytes();collect(json.loads(data),prompts,seeds)
        history[name]=hashlib.sha256(data).hexdigest()
    assert len(PROMPTS)==104 and len(set(PROMPTS))==104,len(PROMPTS)
    assert not ({' '.join(p.lower().split()) for p in PROMPTS}&prompts)
    rng=random.Random(2026112402);pool=list(PROMPTS);rng.shuffle(pool)
    works=[]
    for i,p in enumerate(pool):
        split='CAL' if i<24 else 'TEST';n=i if i<24 else i-24
        seed=rng.randrange(2**31)
        assert seed not in seeds
        seeds.add(seed)
        reg=rng.sample(range(16),3);foreign=rng.sample(range(16,32),3)
        arms=['ordinary','coarse'];rng.shuffle(arms)
        near=max(prompts,key=lambda x:SequenceMatcher(None,p.lower(),x).ratio())
        works.append(dict(id=f'{split}-{n:03}',split=split,prompt=p,seed=seed,reg=reg,foreign=foreign,
                          arm_order=arms,nearest_prior_prompt=near,similarity=SequenceMatcher(None,p.lower(),near).ratio()))
    audit=ROOT.parent/'audits/matched-loss-dev-readonly-20261124'
    summary=json.loads((audit/'summary.json').read_text())
    assert summary['status']=='DEV_MATCHING_GATE_MET' and summary['selected_gain']==1.025
    budget=dict(historical_resources=summary['cumulative_resources'],historical_debits=summary['cumulative_budget_debits'],
                family_caps_seconds=dict(preparation=1080,model=5400,settlement=60),
                dev_stage_costs=summary['stages'])
    used=dict(preparation=summary['stages']['preparation']['wall_seconds'],
              model=summary['stages']['model']['wall_seconds'],
              settlement=summary['stages']['settlement_and_entry_tail']['wall_seconds'])
    budget['proposed_wall_caps_seconds']={k:cap-used[k] for k,cap in budget['family_caps_seconds'].items()}
    value=dict(master_seed=2026112402,selected_gain=1.025,coarse_transfer=.1,codes=dev['codes'],works=works,
               history=history,historical_prompt_count=len(prompts),budget=budget,
               checkpoint_history='UNVERIFIED',unseen_content='USER_ACCEPTED_ASSUMPTION',
               source='104 manually authored scenes; one shuffle and seed/identity/order draw; no generation before freeze')
    TARGET.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    (ROOT/'matched_compare/dev_summary.json').write_bytes((audit/'summary.json').read_bytes())
    (ROOT/'matched_compare/dev_quality.json').write_bytes((audit/'quality_records.json').read_bytes())
    print('FROZEN',len(works),'prior',len(prompts),'max_similarity',max(w['similarity'] for w in works))

if __name__=='__main__':main()
