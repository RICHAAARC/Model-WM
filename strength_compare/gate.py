"""Standard-library gate: no torch/scipy dependency before environment preparation."""
import json
import math
from pathlib import Path

PACKAGE=Path(__file__).resolve().parent
GAINS=dict(low=1.025,near=1.030,high=1.040)

def load(name):return json.loads((PACKAGE/name).read_text())

def validate_dev(summary,quality,allocation):
    expected=load('dev_summary.json')
    if (summary.get('status')!='DEV_BRACKET_GATE_MET' or
        summary.get('selected_gains')!=GAINS or
        summary.get('resources_final') is not True or summary.get('unknown_tail') or
        summary.get('retry') is not False or summary.get('cal_test_started') is not False or
        summary.get('holdout_unsealed') is not False or summary.get('training_updates')!=0):
        raise ValueError('DEV is incomplete, consumed beyond DEV, or changed')
    for kind,n in [('generate',8),('write',336),('read',0)]:
        if summary.get('operations',{}).get(kind)!={'started':n,'completed':n,'denominator':n}:
            raise ValueError('DEV operation family changed')
    if quality!=load('dev_quality.json') or allocation!=load('allocation.json'):
        raise ValueError('DEV quality/allocation differs from inspected frozen evidence')
    if summary.get('geometric_loss_ratios')!=expected['geometric_loss_ratios']:
        raise ValueError('DEV selection changed')
    if load('dev_visual_review.json')['status']!='ACCEPTABLE_FOR_FROZEN_FORMAL_PREPARATION':
        raise ValueError('DEV image review not completed')
    for key in ('cumulative_resources','cumulative_budget_debits'):
        if set(summary[key])!=set(expected[key]) or any(not isinstance(v,(int,float)) or not math.isfinite(v) or v<0 for v in summary[key].values()):
            raise ValueError('Invalid inherited resource ledger')
    return GAINS.copy()
