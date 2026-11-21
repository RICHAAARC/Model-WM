"""Encoding-side prediction and blinded scoring contracts. No model dependencies."""
from itertools import combinations
from math import comb
import numpy as np
from probe_selection.rules import signs

LAYOUTS=('H0','H1','H2','H3')
TRIPLES=np.array(list(combinations(range(16),3)))
EYE=np.eye(48)


def score(q, bank, transform=None):
    x=np.asarray(q,dtype=float)
    if x.shape[-1]!=48 or not np.isfinite(x).all():
        raise ValueError('Invalid logits')
    if transform is not None:
        x=x@transform
    return x@bank[:16].T/48


def margins(pred,groups):
    groups=np.asarray(groups)
    rows=np.arange(len(groups))[:,None]
    truth=pred[rows,groups].min(axis=1)
    others=pred.copy()
    others[rows,groups]=-np.inf
    return truth-others.max(axis=1)


def fit(rows,banks):
    """DEV JPEG85 registered mixture rows only. Truth enters offline design."""
    if not rows or any(not r['work'].startswith('DEV-') or r['book'] not in LAYOUTS for r in rows):
        raise ValueError('Expected geometry-family DEV rows')
    bits=[]; reads=[]; levels=[]; actual=[]
    for r in rows:
        bank=banks[r['book']]
        for n,g in enumerate(r['groups'][:2]):
            bits.extend(bank[g])
            reads.extend(r['reads'][f'u{i}']['jpeg85'] for i in g)
            levels.append(bank[g].mean(axis=0))
            actual.append(r['reads'][f'm{n}']['jpeg85'])
    x,q,z,y=map(np.asarray,(bits,reads,levels,actual))
    xm,qm=x.mean(axis=0),q.mean(axis=0)
    xc=x-xm
    slopes=np.linalg.solve(xc.T@xc+4*EYE,xc.T@(q-qm))
    intercept=qm-xm@slopes
    e=q-(intercept+x@slopes)
    cov=e.T@e/len(e)
    scale=np.sqrt(np.maximum(cov.diagonal(),1e-12))
    corr=np.abs(cov/np.outer(scale,scale))
    np.fill_diagonal(corr,-np.inf)
    neighbor=corr.argmax(axis=1)
    val,vec=np.linalg.eigh(cov)
    white=(vec*(1/np.sqrt(np.maximum(val,0)+max(np.trace(cov)/48,.000001))))@vec.T
    residual=y-(intercept+z@slopes)
    bins=np.rint((z+1)*1.5).astype(int)
    disagreement=np.abs(z[:,neighbor])<.5
    means=np.zeros((4,48)); rate=np.zeros_like(means)
    for k in range(4):
        mask=bins==k
        count=mask.sum(axis=0)
        means[k]=(mask*residual).sum(axis=0)/(4+count)
        rate[k]=((mask*disagreement).sum(axis=0)+2)/(4+count)
    col=np.arange(48)
    feature=z*(disagreement-rate[bins,col])
    # Residualize against all observed within-bit configuration means.
    centers=np.zeros_like(means)
    for k in range(4):
        mask=bins==k
        centers[k]=(mask*feature).sum(axis=0)/np.maximum(mask.sum(axis=0),1)
    feature-=centers[bins,col]
    energy=(feature**2).sum()
    beta=(feature*(residual-means[bins,col])).sum()/(energy+4)
    return dict(intercept=intercept,slopes=slopes,neighbor=neighbor,white=white,means=means,
                rate=rate,centers=centers,beta=beta,energy=energy)


def predict(model,bank,groups,cross):
    z=bank[groups].mean(axis=1)
    bins=np.rint((z+1)*1.5).astype(int)
    col=np.arange(48)
    q=model['intercept']+z@model['slopes']+model['means'][bins,col]
    f=z*((np.abs(z[:,model['neighbor']])<.5)-model['rate'][bins,col])-model['centers'][bins,col]
    return q+(model['beta']*f if cross else 0)


def expected(model,bank,groups,cross,white=False):
    q=predict(model,bank,groups,cross)
    return margins(score(q,bank,model['white'] if white else None),groups).mean()


def measured(row,bank,transform=None):
    q=[row['reads'][f'm{i}']['jpeg85'] for i in range(2)]
    return margins(score(q,bank,transform),row['groups'][:2]).mean()


def dev_decision(records,allocation):
    banks={k:signs(v) for k,v in allocation['books'].items()}
    dev=[w for w in allocation['works'] if w['split']=='DEV']
    lookup={(r['work'],r['book']):r for r in records}
    if len(records)!=48 or set(lookup)!={(w['id'],k) for w in dev for k in banks}:
        raise ValueError('Missing or duplicate DEV rows')
    for w in dev:
        for book in banks:
            r=lookup[w['id'],book]
            if r['groups']!=w['groups']:
                raise ValueError('Identity pairing mismatch')
            keys={f'u{i}' for group in w['groups'] for i in group}|{'m0','m1','m2','clean'}
            if set(r['reads'])!=keys or np.shape(r['dist'])!=(9,) or not np.isfinite(r['dist']).all():
                raise ValueError('Incomplete query or quality family')
            for q in r['reads'].values():
                if set(q)!={'png','jpeg85'}:
                    raise ValueError('Incomplete codec family')
                for x in q.values():
                    if np.shape(x)!=(48,) or not np.isfinite(x).all():
                        raise ValueError('Malformed readout')
    rows=[r for r in records if r['book'] in LAYOUTS]
    work_wins=[]
    for w in dev:
        wins=[0,0,0]
        for left,right in combinations(LAYOUTS,2):
            # Every triple is unique across works in the allocation.
            train=[r for r in rows if r['work']!=w['id'] and r['book'] not in (left,right)]
            model=fit(train,banks)
            pair=[lookup[w['id'],k] for k in (left,right)]
            delta=measured(pair[0],banks[left])-measured(pair[1],banks[right])
            truth=0 if abs(delta)<=1e-10 else np.sign(delta)
            for i,(cross,white) in enumerate(((False,False),(False,True),(True,False))):
                a,b=[expected(model,banks[k],np.array(w['groups'][:2]),cross,white) for k in (left,right)]
                wins[i]+=int(truth!=0 and abs(a-b)>1e-10 and np.sign(a-b)==truth)
        work_wins.append(wins)
    full=fit(rows,banks)
    halves=[fit([r for r in rows if int(r['work'][-2:])%2==i],banks) for i in range(2)]
    stable=int((halves[0]['neighbor']==halves[1]['neighbor']).sum())
    def select(values):
        return max(LAYOUTS,key=values.__getitem__)
    choices={label:select({k:expected(full,banks[k],TRIPLES,cross,white) for k in LAYOUTS})
             for label,cross,white in [('ordinary',False,False),('candidate',True,False),('whitened',False,True)]}
    choices['ideal']='H0'
    choices['empirical']=select({k:np.mean([measured(r,banks[k]) for r in rows if r['book']==k]) for k in LAYOUTS})
    wins=sum(c>max(a,b) for a,b,c in work_wins)
    # Best ordinary physical book, ranked by observed mean native margin; ties follow arm order.
    controls=[choices[n] for n in ('ordinary','whitened','empirical','ideal')]+['random','tardos']
    def value(k):
        return np.mean([measured(r,banks[k]) for r in records if r['book']==k])
    strongest=max(controls,key=value)
    go=(wins>=6 and stable>=12 and full['energy']>1e-8 and choices['candidate'] not in
        {choices['ordinary'],strongest,choices['empirical']})
    return dict(paired_order_counts=work_wins,works_improved=wins,stable_coordinates=stable,
                interaction_energy=float(full['energy']),choices=choices,strongest_native_control=strongest,
                eligible_for_separate_CAL_authorization=bool(go))


def native_or_tardos(q,bank,p=None):
    if p is None: return score(q,bank)
    # Symmetric Tardos simple decoder with the sign of each received logit.
    q=np.asarray(q)
    if q.shape!=(48,) or not np.isfinite(q).all(): raise ValueError('Invalid logits')
    b=(bank[:16]+1)/2
    return ((2*(q>0)-1)*(b-p)/np.sqrt(p*(1-p))).mean(axis=1)


def calibrate(families):
    if len(families)!=24 or any(not family for family in families):
        raise ValueError('Expected 24 nonempty work families')
    maxima=[]
    for family in families:
        tail=[]
        for scores,members in family:
            s=np.asarray(scores)
            if s.shape!=(16,) or not np.isfinite(s).all(): raise ValueError('Invalid score vector')
            tail.extend(s[[i for i in range(16) if i not in members]])
        maxima.append(max(tail))
    return max(maxima)


def paired(candidate,control):
    if len(candidate)!=16 or len(control)!=16: raise ValueError('Expected sixteen work pairs')
    wins=sum(a and not b for a,b in zip(candidate,control))
    losses=sum(b and not a for a,b in zip(candidate,control))
    n=wins+losses
    return wins-losses,sum(comb(n,k) for k in range(wins,n+1))/2**n


def adjudicate(scores,threshold,members):
    s=np.asarray(scores)
    if s.shape!=(16,) or not np.isfinite(s).all() or not np.isfinite(threshold):
        raise ValueError('Invalid frozen decision inputs')
    returned=set(np.flatnonzero(s>threshold))
    wrong=returned-set(members)
    return dict(returned=sorted(returned),wrong=sorted(wrong),reject=not returned,
                safe=bool(returned) and not wrong,
                all_safe=bool(members) and returned==set(members))
