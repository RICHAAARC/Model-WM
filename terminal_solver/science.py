from statistics import mean,median
from latent_control.science import negative_max, calibrate
from .protocol import GAINS

def positives(bundle):
    rows=[r for r in bundle['rows'] if r['kind']=='registered_mix']
    if not bundle['complete'] or len(rows)!=6:
        raise ValueError('Incomplete registered mixture family')
    return [max(r['scores'][j] for j in r['truth']) for r in rows]

def select(dev):
    ranks={}
    margins={}
    for arm in 'CO':
        margins[arm]={}
        for g in GAINS:
            bundles=dev[arm][g]
            if len(bundles)!=8 or len({b['id'] for b in bundles})!=8:
                raise ValueError('DEV denominator mismatch')
            margins[arm][g]=median(min(positives(b))-negative_max(b) for b in bundles)
        ordered=sorted(GAINS,key=lambda g:(-margins[arm][g],g))
        ranks[arm]={g:i+1 for i,g in enumerate(ordered)}
    metrics={a:{g:{k:mean(b['proxies'][k] for b in dev[a][g]) for k in ('layout','artifact')}
                   for g in GAINS} for a in 'CO'}
    def eligible(arm,g,anchor):
        return all(metrics[arm][g][k]<=anchor[k] for k in ('layout','artifact'))
    ordinary=min((g for g in GAINS if eligible('O',g,metrics['O'][1])), key=ranks['O'].get)
    feasible=[g for g in GAINS if eligible('C',g,metrics['O'][ordinary])]
    grid=[dict(C=c,O=o,C_no_worse=eligible('C',c,metrics['O'][o]),
               O_within_reference=eligible('O',o,metrics['O'][1])) for c in GAINS for o in GAINS]
    gains={'O':ordinary,'C':min(feasible,key=ranks['C'].get)} if feasible else None
    return dict(status='SELECTED' if gains else 'STOP_DEV', margins=margins,grid=grid,
                proxies=metrics,gains=gains)

def verdict(test,cal):
    from scipy.stats import binomtest
    ids=[b['id'] for b in test['C']]
    if len(set(ids))!=64 or ids!=[b['id'] for b in test['O']] or len(ids)!=64:
        raise ValueError('TEST pairing')
    pairs=[]
    for i,wid in enumerate(ids):
        row={'id':wid}
        for a in 'CO':
            b=test[a][i]
            threshold=cal[a]['threshold']
            ok=all(x>threshold for x in positives(b))
            negative_max(b)
            errors={r['name']:[j for j,s in enumerate(r['scores']) if s>threshold and j not in r['truth']] for r in b['rows']}
            row[a]=dict(success=ok,catch=ok,proxies={k:b['proxies'][k] for k in ('layout','artifact')},
                        errors={k:v for k,v in errors.items() if v})
        pairs.append(row)
    wins=sum(r['C']['success'] and not r['O']['success'] for r in pairs)
    losses=sum(r['O']['success'] and not r['C']['success'] for r in pairs)
    both=sum(r['C']['success'] and r['O']['success'] for r in pairs)
    p=binomtest(wins,wins+losses,.5,alternative='greater').pvalue if wins+losses else 1.
    wc,lc=[binomtest(k,64).proportion_ci(.975) for k in (wins,losses)]
    means={a:{k:mean(r[a]['proxies'][k] for r in pairs) for k in ('layout','artifact')} for a in 'CO'}
    differences={k:means['C'][k]-means['O'][k] for k in means['C']}
    risks=sum(bool(r['C']['errors'] or r['O']['errors']) for r in pairs)
    quality_ok=all(diff<=0 for diff in differences.values())
    tracking_ok=wins+both>=48 and wins-losses>=8 and p<=.05
    passed=quality_ok and tracking_ok and risks==0
    return dict(passed=bool(passed),paired=pairs,wins=wins,losses=losses,both=both,neither=64-wins-losses-both,
                delta=(wins-losses)/64,interval=[wc.low-lc.high,wc.high-lc.low],p=float(p),risk_works=risks,
                tracking_gate=bool(tracking_ok),risk_gate=risks==0,proxy_gate=quality_ok,
                proxy_means=means,paired_proxy_mean_differences=differences,human_review="NOT_PERFORMED")
