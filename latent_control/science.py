"""Blind decisions; truth is used only afterwards for calibration/adjudication."""
import math
import numpy as np
from scipy.stats import binomtest,beta
from .carrier import quality_ok

def accuse(q,registered_codes,threshold):
    q=np.asarray(q,dtype=np.float64);s=np.asarray(registered_codes,dtype=np.float64)@q
    if not np.isfinite(s).all():raise RuntimeError('Invalid continuous evidence')
    return np.flatnonzero(s>threshold).tolist(),s.tolist()

def negative_max(bundle):
    if len(bundle['rows'])!=22 or not bundle['complete']:raise RuntimeError('Incomplete CAL risk family')
    vals=[]
    for row in bundle['rows']:
        vals.extend(s for j,s in enumerate(row['scores']) if j not in row['truth'])
    return max(vals)

def calibrate(bundles,expected=64):
    if len(bundles)!=expected or len({b['id'] for b in bundles})!=expected:raise RuntimeError('CAL denominator mismatch')
    maxima=[negative_max(b) for b in bundles]
    return dict(sealed=True,denominator=expected,threshold=max(maxima),work_maxima=maxima,
        risk_family=['null','foreign_native','foreign_mix','registered_native_nonowner','registered_mix_nonmember'],
        codecs=['png','jpeg85'],rule='strict > all-work all-family CAL maximum')

def evaluate(bundle,threshold,codes,m):
    calls={};false=False;success=True;positive=0;certificates=[]
    for row in bundle['rows']:
        got,_=accuse(row['q'],codes,threshold);calls[row['name']]=got
        false |= bool(set(got)-set(row['truth']))
        if row['kind']=='registered_mix':
            positive+=1;success &= bool(set(got)&set(row['truth']))
            ideal=np.asarray(row['ideal_q']);error=np.asarray(row['q'])-ideal
            delta=float(np.max(np.abs(np.asarray(codes)@error)))
            certificates.append(dict(name=row['name'],delta=delta,
                valid=delta<threshold<max(row['weights'])-delta))
    eligible=bundle['complete'] and all(quality_ok(q,m) for q in bundle['quality'])
    return dict(id=bundle['id'],catch_all=bool(success and positive==6 and eligible),
        false_attribution=bool(false),complete=bundle['complete'],returned=calls,quality_eligible=eligible,
        certificates=certificates)

def decide(evaluations,bundles,costs,m,expected=64):
    if any(len(evaluations[a])!=expected for a in 'BC'):raise RuntimeError('TEST incomplete, no verdict')
    b={r['id']:r for r in evaluations['B']};c={r['id']:r for r in evaluations['C']}
    if set(b)!=set(c) or len(c)!=expected:raise RuntimeError('Pairing mismatch')
    wins=sum(c[k]['catch_all'] and not b[k]['catch_all'] for k in c)
    losses=sum(b[k]['catch_all'] and not c[k]['catch_all'] for k in c)
    p=float(binomtest(wins,wins+losses,.5,alternative='greater').pvalue) if wins+losses else 1.
    hits={a:sum(x['catch_all'] for x in evaluations[a]) for a in 'BC'}
    false={a:sum(x['false_attribution'] for x in evaluations[a]) for a in 'BC'}
    qualities={}
    for arm in 'BC':
        rows=[q for bnd in bundles[arm] for q in bnd['quality']]
        # Fixed 256-copy quality denominator; missing values do not improve a mean.
        complete=len(rows)==expected*4 and all(x['complete'] for x in bundles[arm])
        mse=[q['mse'] for q in rows];ssim=[q['ssim'] for q in rows]
        qualities[arm]=dict(complete=complete,copies=len(rows),mean_mse=float(np.mean(mse)) if mse else None,
            mean_ssim=float(np.mean(ssim)) if ssim else None,
            mean_psnr_db=float(np.mean([-10*math.log10(v) for v in mse])) if mse and all(v>0 for v in mse) else None,
            has_infinite_psnr=any(v==0 for v in mse))
    qb,qc=qualities['B'],qualities['C'];g=m['gates']
    # Compare means with explicit infinities, not JSON NaN; B infinity cannot be beaten by finite C.
    def ps(q):return math.inf if q['has_infinite_psnr'] else (q['mean_psnr_db'] if q['mean_psnr_db'] is not None else -math.inf)
    quality_gate=bool(qb['complete'] and qc['complete'] and ps(qc)>=ps(qb)-g['psnr_noninferiority_db']
        and qc['mean_ssim']>=qb['mean_ssim']-g['ssim_noninferiority'])
    cost_gate=all(costs['C'][k]<=costs['B'][k] for k in ['wall_seconds','cpu_seconds'])
    effect=hits['C']>=g['joint_catch_min'] and wins-losses>=g['net_gain_min'] and p<=g['paired_p_max']
    complete=all(x['complete'] for a in 'BC' for x in evaluations[a])
    numeric=bool(complete and effect and false['C']==0 and quality_gate and cost_gate)
    cert_count=sum(all(z['valid'] for z in r['certificates']) and len(r['certificates'])==6 for r in evaluations['C'])
    return dict(status='NUMERIC_GATES_PASSED_VISUAL_PENDING' if numeric else 'C_NOT_SUPPORTED',numeric_pass=numeric,
        scientific_pass=None,visual_review='UNRESOLVED',innovation_pass=False,denominator=expected,hits=hits,
        false_attribution=false,risk_upper95={a:float(beta.ppf(.95,false[a]+1,expected-false[a])) if false[a]<expected else 1. for a in 'BC'},
        paired=dict(wins=wins,losses=losses,net=wins-losses,p_one_sided=p),quality=qualities,
        gates=dict(complete=complete,effect=effect,risk=false['C']==0,quality=quality_gate,cost=cost_gate),
        costs=costs,certificate_work_count=cert_count,
        mechanism_evidence='Endpoint/codec errors and response spectra saved; causal interpretation requires review',
        claim='Bounded 16-registered/64-identity, K2, T1 mechanism probe only; no universal FWER or novelty PASS')
