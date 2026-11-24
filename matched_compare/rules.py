"""Frozen, work-level matched-loss comparison; no model imports."""
import math
from statistics import mean, stdev
from scipy.stats import binomtest, t
from probe_selection import rules as R

ARMS=('ordinary','coarse')
LIMITS=dict(generate=104,write=1248,read=5200,mix=1248)
PRIMARY='reg/equal3/png'

def validate_raw(work,raw):
    expected={f'{g}/{v}/{c}' for g in ('reg','foreign')
              for v in ('single0','single1','single2','equal2','equal3','unequal3')
              for c in ('png','jpeg85')}|{'null/clean/png','null/clean/jpeg85'}
    if set(raw)!=set(ARMS): raise ValueError('Missing arm')
    for rows in raw.values():
        if set(rows)!=expected: raise ValueError('Missing query')
        if any(len(v)!=16 or not all(math.isfinite(x) for x in v) for v in rows.values()):
            raise ValueError('Invalid scores')
    if len(set(work['reg']))!=3 or len(set(work['foreign']))!=3: raise ValueError('Invalid identity triples')

def calibrate(rows):
    if len(rows)!=24 or len({r['id'] for r in rows})!=24: raise ValueError('Need all 24 CAL works')
    maxima=[]
    for r in rows:
        if r['split']!='CAL': raise ValueError('Non-CAL threshold input')
        validate_raw(r,r['raw'])
        maxima.append(max(R.negative(r,r['raw'][a]) for a in ARMS))
    return dict(tau=max(maxima),work_maxima=maxima,cal_ids=[r['id'] for r in rows],
                n_cal=24,comparison='strict greater',rank_resolution=1/25,
                query_family='all 26 queries per arm, all innocent registered identities')

def evaluate(r,tau):
    validate_raw(r,r['raw'])
    arms={a:R.evaluate(r,r['raw'][a],tau) for a in ARMS}
    pairs={}
    for q in arms['ordinary']:
        o,c=arms['ordinary'][q],arms['coarse'][q]
        pairs[q]=dict(gained=sorted(set(c['caught'])-set(o['caught'])),
                      lost=sorted(set(o['caught'])-set(c['caught'])),
                      common=sorted(set(o['caught'])&set(c['caught'])),
                      new_false=sorted(set(c['false'])-set(o['false'])),
                      common_false=sorted(set(c['false'])&set(o['false'])),
                      d=len(c['caught'])-len(o['caught']),safe_hit_delta=int(c['hit'])-int(o['hit']))
    return dict(id=r['id'],arms=arms,pairs=pairs,quality=r['quality'])

def adjudicate(reports):
    if len(reports)!=80 or len({r['id'] for r in reports})!=80: raise ValueError('Need all 80 TEST works')
    d=[r['pairs'][PRIMARY]['d'] for r in reports]
    w,l=sum(x>0 for x in d),sum(x<0 for x in d)
    p=float(binomtest(w,w+l,.5,alternative='greater').pvalue) if w+l else 1.
    ratios=[math.log(r['quality']['ordinary']/r['quality']['coarse']) for r in reports]
    center=mean(ratios);half=float(t.ppf(.95,79))*stdev(ratios)/math.sqrt(80)
    bounds=[center-half,center+half]
    quality=math.log(.98)<=bounds[0] and bounds[1]<=math.log(1.02)
    new_false=[dict(work=r['id'],query=q,ids=v['new_false']) for r in reports
               for q,v in r['pairs'].items() if v['new_false']]
    gates=dict(quality_proxy_equivalent=quality,paired_direction_p_le_05=p<=.05,
               positive_net_members=sum(d)>0,no_new_false_attribution=not new_false)
    absolute={a:dict(false_works=sum(any(v['false'] for v in r['arms'][a].values()) for r in reports),
                    false_query_identity_total=sum(len(v['false']) for r in reports for v in r['arms'][a].values()),
                    primary_safe_hits=sum(r['arms'][a][PRIMARY]['hit'] for r in reports),
                    primary_rejections=sum(r['arms'][a][PRIMARY]['rejected'] for r in reports)) for a in ARMS}
    return dict(status='NUMERIC_GATES_MET_PENDING_VISUAL_REVIEW' if all(gates.values()) else 'MATCHED_COMPARISON_NOT_SUPPORTED',
                scientific_pass=None,gates=gates,denominator=80,primary=PRIMARY,
                wins=w,losses=l,ties=80-w-l,net_members=sum(d),mean_net_members=mean(d),paired_d=d,
                one_sided_exact_sign_p=p,quality_log_ratio_90ci=bounds,
                quality_ratio_90ci=[math.exp(x) for x in bounds],quality_geometric_ratio=math.exp(center),
                work_quality_ratios=[math.exp(x) for x in ratios],new_false=new_false,absolute=absolute,
                jpeg85_net_members=sum(r['pairs']['reg/equal3/jpeg85']['d'] for r in reports),
                visual_review='PENDING',fwer_certified=False,innovation_pass=False)
