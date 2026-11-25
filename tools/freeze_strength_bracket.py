"""One fixed content/identity draw. Refuses any redraw of the allocation."""
from pathlib import Path
import json,hashlib,random
from difflib import SequenceMatcher
from tools.freeze_matched_compare import collect
ROOT=Path(__file__).resolve().parents[1];P=ROOT/'strength_bracket'
# 192 authored subject descriptions, with 12 neutral photographic settings.
SUBJECTS='''a brass pinhole sundial|a blue ceramic yarn bowl|a folded leather falconry hood|a wooden apple corer|a glass storm lantern chimney|a red enamel egg slicer|a hand carved maple door knocker|a copper olive pitter|a miniature wrought iron garden gate|a coiled woven horsehair belt|a jade colored porcelain tea pet|a cast aluminum lemon press
a small walnut kalimba|a polished steel jeweler mandrel|a stack of slate plant labels|a purple blown glass paperweight|a carved horn hair comb|a pewter nutmeg grater|a wooden tambour embroidery hoop|a turquoise ceramic toothbrush holder|a brass spindle of green silk thread|a black enamel optical shutter|a ceramic honey dipper jar|a folded canvas tool wrap
a leather covered binocular case|a porcelain electrical junction rosette|a wooden seed spacing ruler|a red bakelite telephone handset|a copper shoe warming bottle|a small glass butter mold|a hand forged drawknife|a striped woven saddle cinch|a cobalt blue ceramic taper holder|a miniature brass diving helmet|a silver sugar sifter|a polished agate burnishing stone
a grey felt millinery cone|a wooden sock darning mushroom|a golden brass pendulum weight|a wicker bottle carrier|a glazed ceramic fermentation weight|a cast iron seam roller|a spherical glass aeration stone display|a pair of wooden glove stretchers|a copper fountain pen tray|a cream ceramic match striker|a folded green brocade purse|a carved stone mortar pestle
a bronze mechanical pencil sharpener|a clear glass microscope prism|a ceramic clay extruder die|a hand stitched leather luggage tag|a white marble soap dish|a brass hose spray nozzle|a shallow wooden paint pigment tray|a yellow glazed egg coddler|a pair of steel watchmaker tweezers|a silver plated toast rack|a small wooden pulley block|a green glass flower frog
a blue enameled metal thimble|a wooden leather stitching pony|a copper laboratory scoop|a round wool felt polishing pad|a pale ceramic oil lamp|a small steel fishing reel spool|a purple silk tassel bundle|a brass botanical specimen press|a folded flax shopping net|a porcelain dental shade guide|a wooden spinning top launcher|a bronze decorative drawer pull
a brown glazed bean pot|a clear glass slide staining rack|a pair of cast iron shutter hinges|a polished walnut needle case|a rolled copper sheet sample|a white stone incense pebble tray|a leather archery finger tab|a cream enamel funnel|a brass rope fid|a small bamboo tea whisk holder|a blue glazed bonsai drainage screen|a silver plated berry spoon
a wooden trumpet mute|a black steel mandolin bridge|a red ceramic pipe rest|a polished quartz egg on a stand|a folded herringbone wool scarf|a copper piping tip organizer|a pale green porcelain pickle dish|a round tin tea sieve|a wooden pastry crimper|a bronze hourglass frame|a leather camera wrist strap|a ceramic citrus zester plate
a carved maple music box cylinder|a pair of silver knitting stitch holders|a clear glass vacuum flask insert|a small steel rivet setter|a gold glazed ceramic rabbit figurine|a folded canvas binocular sling|a dark wooden bookbinding press|a brass compass divider|a turquoise glass perfume atomizer|a woven straw panama hat band|a ceramic mushroom cultivation collar|a walnut handled cheese plane
a white porcelain laboratory spot plate|a brass beekeeping queen cage|a terracotta bird nesting cup|a green enamel soap shaker|a single polished horn button|a copper violin chinrest fitting|a wooden weaving tablet set|a blue ceramic vase collar|a cast iron cobbler heel iron|a pale stone scroll weight|a red lacquered ink brush rest|a steel leather edge beveler
a black walnut yarn swift hub|a folded silk pocket square|a white ceramic orchid clip holder|a brass leather stamping wheel|a small glass battery jar|a bronze model cannon carriage|a green glazed ceramic bonsai pot|a pair of cork shoe lasts|a copper pouring lip attachment|a carved wooden reed case|a yellow porcelain butter stamp|a blue steel watch spring housing
a turned maple honey spoon|a grey ceramic shaving mug|a red metal tape dispenser|a brass needle threader|a small cut glass celery vase|a folded brown leather map case|a silver wire basket liner|a wooden bicycle rim section|a copper distillation condenser coil|a white glazed ink palette|a miniature stone grinding wheel|a pair of green fabric shoe gaiters
a clear glass seed germination jar|a wooden hand loom comb|a brass candle molding tube|a pale jade ceramic spoon rest|a black iron window stay|a folded purple silk fan sleeve|a small tin lantern reflector|a glazed terracotta salt cellar|a steel clay ribbon cutter|a carved ivory colored resin die|a bronze bell clapper|a leather fountain pen sleeve
a copper hand pump cylinder|a white ceramic shaving brush stand|a round maple checkers board|a blue glass mixing beaker|a yellow enamel weighing scoop|a polished steel bone folder|a folded linen bread bag|a brass gasket cutting compass|a ceramic nesting bird whistle|a wooden espresso tamping block|a red lacquered chopstick case|a small bronze door escutcheon
a silver plated lemon fork|a green glass seed bead vial|a steel sailmaker palm tool|a wooden leather burnishing wheel|a cobalt ceramic sake cup|a rolled bamboo sushi mat|a brass clock winding key|a stoneware mushroom shaped jar|a black enamel laboratory clamp|a white shell inlay blank|a copper coffee filter basket|a folded blue denim apron
a miniature wooden grain elevator|a red glazed ceramic chimney pot model|a glass wickless alcohol burner|a brass garden hose splitter|a carved walnut picture frame corner|a small steel knot tying jig|a cream ceramic tea bag caddy|a woven reed fish basket lid|a bronze miniature sundial compass|a silver plated candle drip pan|a pale blue glass dessert pedestal|a wooden cord winding shuttle'''.replace('\n','|').split('|')
SETTINGS=('on a plain grey table in soft side light','on an aged pine workbench with a cream wall behind it','on a dark felt mat beside a small window','on a pale stone shelf under diffuse daylight','on a linen covered table with a gently blurred background','on a clean oak counter beneath a narrow ceiling lamp','on a charcoal display plinth with soft shadows','on a weathered cedar surface in overcast daylight','on a neutral beige tabletop with a folded cloth nearby','on a dark slate board against a pale wall','on a simple studio shelf with broad frontal light','on a pale wooden bench beside a frosted window')

def main():
    target=P/'allocation.json'
    if target.exists():raise FileExistsError('Already frozen; no redraw')
    old=json.loads((ROOT/'sd2_matched/allocation.json').read_text())
    paths=list(old['history'])+['sd2_matched/allocation.json']
    prompts=set();seeds=set();history={}
    for name in dict.fromkeys(paths):
        b=(ROOT/name).read_bytes();collect(json.loads(b),prompts,seeds);history[name]=hashlib.sha256(b).hexdigest()
    assert len(SUBJECTS)==192 and len(set(SUBJECTS))==192,len(SUBJECTS)
    pool=[f'A documentary photograph of {subject}, {SETTINGS[i%12]}.' for i,subject in enumerate(SUBJECTS)]
    assert not set(' '.join(p.lower().split()) for p in pool)&prompts
    rng=random.Random(2026112502);rng.shuffle(pool);works=[]
    for i,prompt in enumerate(pool):
        split='DEV' if i<8 else 'CAL' if i<32 else 'TEST';j=i if i<8 else i-8 if i<32 else i-32
        seed=rng.randrange(2**31);assert seed not in seeds;seeds.add(seed)
        reg=rng.sample(range(16),3);foreign=rng.sample(range(16,32),3)
        arms=['coarse','low','near','high'];rng.shuffle(arms)
        actions=['coarse','1.02','1.025','1.03','1.035','1.04','1.045'];rng.shuffle(actions)
        near=max(prompts,key=lambda x:SequenceMatcher(None,prompt.lower(),x).ratio())
        works.append(dict(id=f'{split}-{j:03}',split=split,prompt=prompt,seed=seed,reg=reg,foreign=foreign,arm_order=arms,action_order=actions,nearest_prior_prompt=near,similarity=SequenceMatcher(None,prompt.lower(),near).ratio()))
    previous=json.loads((ROOT.parent/'audits/sd21-compare-readonly-20261125/summary.json').read_text())
    value=dict(master_seed=2026112502,codes=old['codes'],works=works,coarse_transfer=.1,history=history,historical_prompt_count=len(prompts),
      source='192 authored subjects with 12 fixed neutral photographic settings; one shuffle and one seed/identity/order draw; recurrent subject categories allowed',
      checkpoint_history='UNVERIFIED',unseen_content='USER_ACCEPTED_ASSUMPTION',budget=dict(
        historical_resources=previous['cumulative_resources'],historical_debits=previous['cumulative_budget_debits'],hard_time_caps=None,
        estimated_minutes=dict(dev_preparation=[5,20],dev_model=[5,15],formal_model=[150,270]),
        fixed_family_operations=dict(generate=192,write=4752,mix=4416,read=18032),
        dev_operations=dict(generate=8,write=336,mix=0,read=0)))
    target.write_text(json.dumps(value,indent=2)+'\n')
    (P/'prior_summary.json').write_text(json.dumps(previous,indent=2)+'\n')
    print(json.dumps(dict(frozen=len(works),history_prompts=len(prompts),max_similarity=max(w['similarity'] for w in works))))
if __name__=='__main__':main()
