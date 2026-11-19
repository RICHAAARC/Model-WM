"""Shared selector, open-set calibration, and paired adjudication. No carrier internals."""
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

ACTIONS=('native','fine','coarse')
ARMS=('native','global','ordinary','confidence','margin')
PRIMARY='reg/equal3/png'
COMPOSITIONS={'equal2':(.5,.5),'equal3':(1/3,1/3,1/3),'unequal3':(.5,.25,.25)}

def signs(codes):
    return np.array([[2*int(bit)-1 for bit in c] for c in codes],dtype=float)

def scores(q,bank):
    x=np.asarray(q,dtype=float)
    if x.shape!=(bank.shape[1],) or not np.isfinite(x).all():
        raise ValueError('Invalid continuous readout')
    return (bank@x/len(x)).tolist()

def margin(v,ids):
    return max(v[i] for i in ids)-max(s for i,s in enumerate(v) if i not in ids)

def distance(clean,copy):
    def small(x):
        return np.asarray(Image.fromarray(x).resize((128,128),Image.Resampling.BOX))
    return float(1-structural_similarity(small(clean),small(copy),channel_axis=-1,data_range=255))

def mix(images,weights):
    if len(images)!=len(weights):
        raise ValueError('Wrong mixture arity')
    value=sum(w*x.astype(np.float64) for x,w in zip(images,weights))
    return np.rint(value).clip(0,255).astype('uint8')

def choose(values,allowed):
    best=max(values[a] for a in allowed)
    winners=[a for a in allowed if values[a]==best]
    return winners[0] if len(winners)==1 else 'native'

def probe_stat(mixed,singles,triples,bank,quality):
    deltas=[]; errors=[]
    for ids,rows in zip(triples,mixed):
        target=bank[ids].mean(axis=0)
        for q in rows:
            deltas.append(margin(scores(q,bank),ids))
            norm=max(np.linalg.norm(q)*np.linalg.norm(target),1.e-12)
            errors.append(float(np.dot(q,target)/norm))
    confidence=[scores(q,bank)[uid] for uid,q in singles]
    return dict(margin=min(deltas),ordinary=min(errors),confidence=min(confidence),quality=quality)

def pick(probes,global_action,ratio):
    allowed=[a for a in probes if probes[a]['quality']<=ratio*probes['native']['quality']+1.e-12]
    return dict(native='native',**{'global':global_action},
                **{arm:choose({a:probes[a][arm] for a in probes},allowed)
                   for arm in ('ordinary','confidence','margin')})

def labels(work,condition):
    group,view,_=condition.split('/')
    if group!='reg':
        return []
    if view.startswith('single'):
        return [work['reg'][int(view[-1])]]
    return work['reg'][:len(COMPOSITIONS[view])]

def negative(work,rows):
    return max(s for key,v in rows.items() for i,s in enumerate(v) if i not in labels(work,key))

def evaluate(work,rows,tau):
    result={}
    for key,v in rows.items():
        truth=set(labels(work,key))
        returned={i for i,s in enumerate(v) if s>tau}
        result[key]=dict(returned=sorted(returned),caught=sorted(returned & truth),
                         false=sorted(returned-truth),rejected=not returned,
                         all_caught=bool(truth) and truth<=returned,
                         hit=bool(returned & truth) and not returned-truth)
    return result

def verdict(reports):
    def hit(r,arm):return int(r['arms'][arm][PRIMARY]['hit'])
    def false(r,arm):
        return {(key,i) for key,v in r['arms'][arm].items() for i in v['false']}
    if len(reports)!=8:
        raise ValueError('Need eight complete exploration works')
    gains={a:sum(hit(r,'margin')-hit(r,a) for r in reports) for a in ARMS if a!='margin'}
    new_false={a:[r['id'] for r in reports if false(r,'margin')-false(r,a)]
               for a in ('native','global','ordinary')}
    predicted=[r['id'] for r in reports if r['choices']['margin']!='native' and
               r['real_margins'][r['choices']['margin']]>r['real_margins']['native']]
    fixed_hits={a:sum(r['fixed_primary'][a]['hit'] for r in reports) for a in reports[0]['fixed_primary']}
    wins=sum(hit(r,'margin') for r in reports)
    distinct_best=set.intersection(*(set(r['best_actions']) for r in reports))
    quality=all(r['quality_ok'] for r in reports)
    if all(hit(r,'native') for r in reports):
        status='PRIMARY_SATURATED_WITH_FALSE_ATTRIBUTION' if any(false(r,'native') for r in reports) else 'NATIVE_SUFFICIENT'
    elif len(predicted)<2:
        status='CROSS_IDENTITY_PREDICTION_UNSUPPORTED'
    elif wins<=max(fixed_hits.values()) or distinct_best:
        status='FIXED_ACTION_SUFFICIENT' if wins>=sum(hit(r,'native') for r in reports) else 'NO_ATTRIBUTION_GAIN'
    elif (8-sum(hit(r,'native') for r in reports)>=2 and min(gains[a] for a in ('ordinary','global'))>=2
          and not any(new_false.values()) and quality):
        status='NUMERIC_SIGNAL_PENDING_VISUAL_REVIEW'
    else:
        status='NO_QUALIFYING_PAIRED_GAIN'
    return dict(status=status,net=gains,fixed_hits=fixed_hits,new_false_works=new_false,
                directional_transfer=predicted,quality_proxy_pass=quality,visual_review='PENDING',
                second_carrier_authorized=False,scientific_pass=None)
