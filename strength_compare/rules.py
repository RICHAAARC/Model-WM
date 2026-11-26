"""Four-arm calibration and paired reports; frozen bracket verdict is reused unchanged."""
import math
from probe_selection import rules as R
from strength_bracket.rules import ARMS,PRIMARY,adjudicate

LIMITS=dict(generate=184,write=4416,mix=4416,read=18032)
QUERIES={f'{g}/{v}/{c}' for g in ('reg','foreign')
         for v in ('single0','single1','single2','equal2','equal3','unequal3')
         for c in ('png','jpeg85')}|{'null/clean/png','null/clean/jpeg85'}

def validate_raw(work,raw):
    if set(raw)!=set(ARMS):raise ValueError('Missing arm')
    for rows in raw.values():
        if set(rows)!=QUERIES:raise ValueError('Missing query')
        if any(len(v)!=16 or not all(math.isfinite(x) for x in v) for v in rows.values()):
            raise ValueError('Invalid scores')
    if len(set(work['reg']))!=3 or len(set(work['foreign']))!=3 or not all(0<=i<16 for i in work['reg']) or not all(16<=i<32 for i in work['foreign']):
        raise ValueError('Invalid identity triples')
    if set(work['quality'])!=set(ARMS) or any(not math.isfinite(x) or x<=0 for x in work['quality'].values()):
        raise ValueError('Invalid quality')

def calibrate(rows):
    if len(rows)!=24 or len({r['id'] for r in rows})!=24:raise ValueError('Need all 24 CAL works')
    maxima=[]
    for r in rows:
        if r['split']!='CAL':raise ValueError('Non-CAL threshold input')
        validate_raw(r,r['raw'])
        maxima.append(max(R.negative(r,r['raw'][a]) for a in ARMS))
    return dict(tau=max(maxima),work_maxima=maxima,cal_ids=[r['id'] for r in rows],
                n_cal=24,comparison='strict greater',rank_resolution=1/25,
                query_family='all four arms, all 26 queries per arm, all innocent registered identities')

def evaluate(r,tau):
    if r['split']!='TEST' or not math.isfinite(tau):raise ValueError('Invalid TEST input or threshold')
    validate_raw(r,r['raw'])
    arms={a:R.evaluate(r,r['raw'][a],tau) for a in ARMS}
    pairs={}
    for arm in ('low','near','high'):
        pairs[arm]={}
        for q in sorted(QUERIES):
            c,o=arms['coarse'][q],arms[arm][q]
            pairs[arm][q]=dict(gained=sorted(set(c['caught'])-set(o['caught'])),
                lost=sorted(set(o['caught'])-set(c['caught'])),common=sorted(set(o['caught'])&set(c['caught'])),
                coarse_new_false=sorted(set(c['false'])-set(o['false'])),
                ordinary_new_false=sorted(set(o['false'])-set(c['false'])),
                common_false=sorted(set(c['false'])&set(o['false'])),
                d=len(c['caught'])-len(o['caught']),safe_hit_delta=int(c['hit'])-int(o['hit']))
    return dict(id=r['id'],arms=arms,pairs=pairs,quality=r['quality'])

def summarize(reports):
    if len(reports)!=160 or len({r['id'] for r in reports})!=160:raise ValueError('Need all 160 TEST works')
    quality={a:[r['quality'][a]/r['quality']['coarse'] for r in reports] for a in ('low','near','high')}
    errors={a:{kind:[dict(work=r['id'],query=q,ids=v[kind]) for r in reports
                    for q,v in r['pairs'][a].items() if v[kind]]
               for kind in ('coarse_new_false','ordinary_new_false','common_false')}
            for a in ('low','near','high')}
    d=[r['pairs']['high'][PRIMARY]['d'] for r in reports]
    safe=[r['pairs']['high'][PRIMARY]['safe_hit_delta'] for r in reports]
    result=adjudicate(d,safe,bool(errors['high']['coarse_new_false']),bool(errors['high']['ordinary_new_false']),quality)
    absolute={a:{q:dict(true_members=sum(len(r['arms'][a][q]['caught']) for r in reports),
                       safe_captures=sum(r['arms'][a][q]['hit'] for r in reports),
                       false_works=sum(bool(r['arms'][a][q]['false']) for r in reports),
                       false_identity_total=sum(len(r['arms'][a][q]['false']) for r in reports),
                       rejections=sum(r['arms'][a][q]['rejected'] for r in reports),work_denominator=160)
                 for q in sorted(QUERIES)} for a in ARMS}
    secondary={a:{q:dict(wins=sum(r['pairs'][a][q]['d']>0 for r in reports),
                        losses=sum(r['pairs'][a][q]['d']<0 for r in reports),
                        ties=sum(r['pairs'][a][q]['d']==0 for r in reports),
                        net_members=sum(r['pairs'][a][q]['d'] for r in reports))
                  for q in sorted(QUERIES)} for a in ('low','near','high')}
    result.update(paired_d=d,paired_safe_d=safe,work_quality_ratios=quality,errors=errors,
                  absolute=absolute,paired_all_conditions=secondary,primary_control='high',
                  fwer_certified=False,innovation_pass=False)
    return result
