"""Read sealed DEV and carry history forward; never extract or select on logits."""
import json,math
from pathlib import Path
from statistics import mean
PACKAGE=Path(__file__).resolve().parent

def read(p): return json.loads(p.read_text())

def verify_dev(out):
    expected=read(PACKAGE/'dev_summary.json')
    s=read(out/'summary.json')
    if not s.get('resources_final') or s.get('unknown_tail') or s.get('retry'):
        raise ValueError('DEV receipt incomplete or replayed')
    for key in ('status','selected_gain','operations','counts','resources',
                'cumulative_resources','cumulative_budget_debits','interface_numeric_passed',
                'interface_reads','interface_works','matching_gate_passed','cal_test_started'):
        if s.get(key)!=expected[key]: raise ValueError('Sealed DEV changed: '+key)
    if s['status']!='DEV_MATCHING_GATE_MET' or s['selected_gain']!=1.025 or s['cal_test_started']:
        raise ValueError('DEV gate not passed')
    review=read(PACKAGE/'dev_visual_review.json')
    if review['status']!='AI_ENGINEERING_SCREEN_PASSED' or review['works_reviewed']!=8:
        raise ValueError('DEV visual screen missing')
    q=read(out/'quality_records.json')
    if q!=read(PACKAGE/'dev_quality.json'): raise ValueError('DEV quality changed')
    if len(q)!=8 or len({x['work'] for x in q})!=8: raise ValueError('Incomplete DEV')
    logs={str(g):mean(math.log(r['loss'][str(g)]/r['loss']['coarse']) for r in q)
          for g in (1.,1.025,1.05,1.075)}
    if any(not math.isfinite(v) for v in logs.values()): raise ValueError('Invalid DEV loss')
    chosen=min(logs,key=lambda x:(abs(logs[x]),float(x)))
    if float(chosen)!=1.025 or not .98<=math.exp(logs[chosen])<=1.02:
        raise ValueError('DEV quality choice mismatch')
    frozen=read(PACKAGE.parent/'sd2_matched/allocation.json')
    if read(out/'allocation.json')!=frozen: raise ValueError('DEV allocation differs')
    formal=read(PACKAGE/'allocation.json')
    if formal['works']!=[w for w in frozen['works'] if w['split'] in ('CAL','TEST')]:
        raise ValueError('Formal work rows changed')
    if formal['codes']!=frozen['codes']: raise ValueError('Codebook changed')
    return s
