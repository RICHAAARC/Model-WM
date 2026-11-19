"""Fixed service scores and labelled exploratory diagnostics, fit half only."""
import numpy as np

WEIGHTS = (.25,.5,.75)
CODECS = ('png','jpeg85')


def signs(code):
    return np.array([2*int(bit)-1 for bit in code])


def scores(q,bank):
    q = np.asarray(q,dtype=np.float64)
    if q.shape != (48,) or not np.isfinite(q).all():
        raise ValueError('Reader must return 48 finite continuous values')
    return dict(soft=(bank@q/48).tolist(),hard=(bank==(np.where(q>0,1,-1))).mean(1).tolist())


def gap(values, members):
    return max(values[i] for i in members)-max(v for i,v in enumerate(values) if i not in members)


def diagnose(work,raw,codes):
    if work['split']!='fit':
        raise ValueError('Only fit records can enter the decision')
    bank = np.stack([signs(c) for c in codes[:16]])
    expected = {n+'_'+c for c in CODECS for n in ['null',*[f'{g}_{i}' for g in ('reg','foreign') for i in range(2)],
                                   *[f'{g}_mix_{i}' for g in ('reg','foreign') for i in range(3)]]}
    if set(raw)!=expected:
        raise ValueError('Incomplete 22-condition family')
    observed = {name:scores(q,bank) for name,q in raw.items()}
    result = {}
    for rule in ('hard','soft'):
        negatives = []
        for name,row in observed.items():
            members = work['registered'] if name.startswith('reg_mix') else (
                [work['registered'][int(name.split('_')[1])]] if name.startswith('reg_') else [])
            negatives.extend(v for j,v in enumerate(row[rule]) if j not in members)
        ceiling = max(negatives)
        # A within-work diagnostic of separation, not an operational rejection threshold.
        rows = []
        for i,beta in enumerate(WEIGHTS):
            for c in CODECS:
                members = work['registered']
                actual = observed[f'reg_mix_{i}_{c}'][rule]
                endpoint = beta*np.array(raw['reg_0_'+c])+(1-beta)*np.array(raw['reg_1_'+c])
                ref = scores(endpoint,bank)[rule]
                rows.append(dict(condition=f'{i}_{c}',weight=beta,codec=c,actual=max(actual[j] for j in members)-ceiling,
                                 endpoint=max(ref[j] for j in members)-ceiling,rank_gap=gap(actual,members)))
        result[rule] = dict(negative_max=ceiling,rows=rows,all_separated=all(r['actual']>0 for r in rows))
    return dict(id=work['id'],scores=observed,**result)


def decision(fit):
    if [x['id'] for x in fit]!=[f'fit-{i:02}' for i in range(8)]:
        raise ValueError('Expected eight complete fit works in frozen order')
    rescued = {}
    for work in fit:
        for row in work['soft']['rows']:
            if row['actual']<=0<row['endpoint']:
                rescued.setdefault(row['condition'],[]).append(work['id'])
    if all(w['soft']['all_separated'] for w in fit):
        status='SOFT_SUFFICIENT_ON_FIT'
    elif any(len(ids)>=2 for ids in rescued.values()):
        status='C_PILOT_SIGNAL_ON_FIT'
    else:
        status='NO_REPEATED_RESCUABLE_CONDITION'
    return dict(status=status,fit_ids=[w['id'] for w in fit],eligible_by_condition=rescued,
                all_separated={k:sum(w[k]['all_separated'] for w in fit) for k in ('hard','soft')},
                decision_unit='work',operating_threshold=None,training_authorized=False,
                visual_semantics='author review of first two panels pending',scientific_pass=None)
