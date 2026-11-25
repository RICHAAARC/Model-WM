"""One draw of new content, all splits and identities. Refuses overwrite."""
from pathlib import Path
import json,hashlib,random
from difflib import SequenceMatcher
from tools.freeze_matched_compare import collect
ROOT=Path(__file__).resolve().parents[1];P=ROOT/'sd2_matched'
PROMPTS='''A basalt metate holds a handful of dry blue corn beneath a thatched shelter.
A silver tuning fork rests on the felt lid of a piano technician's case.
A green conical fermentation vessel stands on a steel grating in a small brewery.
A woven willow lobster creel rests beside a tidal channel at low water.
A miniature hot air engine turns a brass flywheel on a stone worktop.
A glass weather barometer hangs above a narrow dark oak console.
A single bone sewing awl lies beside a roll of undyed felt.
A hand painted ceramic roof finial rests on a low tiled platform.
A folded bronze mesh fire screen leans against an unlit hearth.
A blue water metering wheel sits in a shallow stone irrigation ditch.
A cluster of dried lotus pods rises from a tall narrow pewter vase.
A mahogany marionette control bar hangs above neatly separated strings.
A red cast iron coffee roaster stands beside a burlap sack in a courtyard.
A round slate cheese board holds three small pear halves and a silver pick.
A cream enamel hospital trolley carries a single covered stainless dish.
A translucent amber resin sphere sits on a triangular obsidian stand.
A restored spinning jenny occupies an empty red brick weaving room.
A salmon colored ceramic umbrella stand holds two closed black parasols.
A small wooden model of a suspension bridge spans a painted blue channel.
A brass optical bench supports two convex lenses and a white projection screen.
A silver fish poacher with its lid open rests on a cold black stove.
A pile of curved terracotta drainage tiles lies beside a vineyard trellis.
A cedar shake sample is clamped beneath a carpenter's marking gauge.
A lacquered paper kite shaped like a moth hangs above a blue folding screen.
A green glass insulator is mounted on a weathered crossarm against clear sky.
A rectangular sand casting mold lies open beside a tin ladle.
A rose colored stoneware butter churn stands on a farmhouse doorstep.
A cork handled fly fishing rod rests across two empty wooden stools.
A carved serpentine bookend supports three plain cloth bound notebooks.
A blue ceramic evaporating dish holds a shallow ring of white crystals.
A freestanding brass coat rail carries a single rust colored raincoat.
A white porcelain radiator valve sits beside a dismantled copper pipe joint.
A small ivory colored harpsichord fills the corner of a sunlit music studio.
A pale green seedling tray contains rows of young purple basil plants.
A flat wooden basket contains several unopened cacao pods.
A large riveted ventilation hood hangs over a vacant smith's anvil.
A circular ceramic bat rests on an idle pottery wheel beside a damp sponge.
A frosted glass pendant light hangs above an empty dark walnut desk.
A pair of antique brass candle snuffers lies on a folded linen napkin.
A short row of white ceramic electric fence insulators stands on a workbench.
A green folding camp oven sits open on a rough stone slab.
A bamboo bird whistle rests beside a shallow bowl of clean water.
A polished nickel sextant lies in an open blue velvet case.
A shallow tray holds a complete red wax seal beside its wooden stamp.
A bentwood sled is parked beside a birch sapling on a snowy slope.
A wooden cooper's croze rests across the rim of an unfinished barrel.
A purple glass oil dispenser stands on a narrow limestone kitchen shelf.
A block of green floral foam supports three long white calla lily stems.
A black accordion camera bellows lies extended beside a ground glass plate.
A large oval laundry dolly stands upright in an empty zinc wash tub.
A single yellow ceramic jester mask rests on a charcoal display plinth.
A copper alembic still sits beneath a dark timber beam in a quiet workshop.
A pair of brass stair carpet rods lies beside a roll of burgundy runner.
A small triangular sailing pennant hangs from a stationary wooden mast.
A curved ivory colored drafting spline rests beside two lead weights.
A stack of round clay bee hive entrance discs sits on a garden wall.
A polished wooden chess clock stands between two empty felt trays.
A white glazed crucible sits inside a cold circular laboratory furnace.
A pair of orange cable reels stands beneath an empty metal gantry.
A green silk pincushion holds an orderly ring of long brass pins.
A carved soapstone incense burner rests beside a folded black paper fan.
A yellow aluminum surveying tripod supports a compact optical level.
A tall earthenware rhubarb forcer stands beside emerging garden shoots.
A leather bound telescope case lies open on a striped wool blanket.
A hand operated sugarcane press stands under an empty corrugated shelter.
A grey felt hat block is mounted on a slender maple pedestal.
A red marine signal lamp hangs above a steel hatch wheel.
A ceramic garlic keeper with round holes sits on a pale beech counter.
A tightly rolled rush floor mat rests against a white plaster pillar.
A black cast iron waffle mold lies open beside a wooden pastry brush.
A pair of pale blue glass door knobs sits on a small square mirror.
A green watering trough supports a single floating white water lily.
A set of nested copper measuring scoops hangs from an iron kitchen rail.
A low round wicker tea table stands beneath a striped canvas awning.
A dismantled mechanical door closer rests on a sheet of brown packing paper.
A large white shell is displayed on a black metal three point stand.
A single maroon bowling pin stands at the end of an empty wooden lane.
A stone mason's pitching chisel lies beside a freshly squared granite block.
A narrow glass chromatography column stands beside an empty receiving flask.
A burnt orange ceramic bread cloche sits open on a baking stone.
A wooden concertina rests on a grey felt cushion in a quiet recital room.
A cream painted seed cabinet contains rows of small empty metal drawers.
A small cast bronze armillary sphere stands beside a still courtyard pond.
A scarlet cloth parasol is folded beside a cylindrical bamboo carrying case.
A polished steel ice cream spade rests beside a chilled empty metal pan.
A tall wire mesh chimney guard stands on a flat grey slate roof.
A single earthenware duck decoy floats beside a weathered reed jetty.
A dark green hand cranked pasta machine is secured to a marble counter.
A shallow woven palm basket holds three rough uncut geodes.
A silver telescopic music stand holds a blank cream card in a rehearsal room.
A carved wooden butter paddle lies beside a pale ceramic mixing bowl.
A green glass demijohn in a rope cradle stands beside a cellar arch.
A black rubber fender hangs between a small boat and a stone quay.
A rectangular wooden marquetry panel displays a geometric star on a worktable.
A pair of copper brazing tongs hangs beside an unused refractory brick hearth.
A blue ceramic rolling pin rests in two wooden supports on a flour dusted slab.
A small circular rain gauge stands in a bed of low alpine plants.
A zinc roof ventilator with a pointed cap rests on a pallet in a yard.
A folded gold paper theater crown lies beside a plain purple costume collar.
A polished brown ceramic ocarina rests on a cushion of pale flax fibers.
A green metal desk perforator stands beside an unmarked stack of cream paper.
A pair of wooden kayak outriggers rests across a narrow pebble beach.
A red fire bell is mounted on a whitewashed wall beside an empty hook.
A small clear acrylic turbine model sits on a black engineering display base.
A coiled strip of pale birch bark lies beside a half finished basket.
A dark blue enamel sauce ladle rests across the rim of an empty copper stockpot.
A stoneware salt pig stands beside a small olive wood salt spoon.
A brass lamp wick trimmer rests in a shallow square tin tray.
A green velvet jewelry bust displays a single polished amber pendant.
A low iron boot scraper stands beside the worn threshold of a stone cottage.
A cream ceramic wall pocket holds a single dried teasel stem.
A mechanical egg grader rests on a clean wooden poultry shed table.'''.splitlines()

def main():
 target=P/'allocation.json'
 if target.exists():raise FileExistsError('Already frozen; no redraw')
 dev=json.loads((ROOT/'matched_loss/allocation.json').read_text())
 paths=list(dev['history'])+['matched_loss/allocation.json','matched_compare/allocation.json']
 prompts=set();seeds=set();history={}
 for name in paths:
  data=(ROOT/name).read_bytes();collect(json.loads(data),prompts,seeds)
  history[name]=hashlib.sha256(data).hexdigest()
 assert len(PROMPTS)==112 and len(set(PROMPTS))==112,len(PROMPTS)
 assert not ({' '.join(p.lower().split()) for p in PROMPTS}&prompts)
 rng=random.Random(2026112501);pool=list(PROMPTS);rng.shuffle(pool);works=[]
 for i,p in enumerate(pool):
  split='DEV' if i<8 else 'CAL' if i<32 else 'TEST';n=i if i<8 else i-8 if i<32 else i-32
  seed=rng.randrange(2**31);assert seed not in seeds;seeds.add(seed)
  reg=rng.sample(range(16),3);foreign=rng.sample(range(16,32),3)
  arms=['ordinary','coarse'];rng.shuffle(arms)
  actions=['coarse','1.0','1.025','1.05','1.075'];rng.shuffle(actions)
  near=max(prompts,key=lambda x:SequenceMatcher(None,p.lower(),x).ratio())
  works.append(dict(id=f'{split}-{n:03}',split=split,prompt=p,seed=seed,reg=reg,foreign=foreign,
      arm_order=arms,action_order=actions,nearest_prior_prompt=near,
      similarity=SequenceMatcher(None,p.lower(),near).ratio()))
 prev=json.loads((ROOT.parent/'audits/matched-compare-readonly-20261125/summary.json').read_text())
 value=dict(master_seed=2026112501,codes=dev['codes'],works=works,coarse_transfer=.1,
     history=history,historical_prompt_count=len(prompts),
     checkpoint_history='UNVERIFIED',unseen_content='USER_ACCEPTED_ASSUMPTION',
     budget=dict(historical_resources=prev['cumulative_resources'],historical_debits=prev['cumulative_budget_debits'],
       hard_time_caps=None,estimated_minutes=dict(dev_preparation=[5,20],dev_model=[5,15],full_model=[70,120]),
       fixed_family_operations=dict(generate=112,write=1488,mix=1248,read=5224)),
     source='112 manually authored scenes; one shuffle and seed/identity/order draw, before generation')
 target.write_text(json.dumps(value,indent=2)+'\n')
 (P/'prior_summary.json').write_text(json.dumps(prev,indent=2)+'\n')
 print('FROZEN',len(works),'prior',len(prompts),'max similarity',max(w['similarity'] for w in works))
if __name__=='__main__':main()
