"""Frozen blind scoring, independent calibration and paired adjudication."""
import math
import numpy as np
from scipy.stats import beta,binomtest
from .protocol import RISK_CONDITIONS,CONDITIONS

def scores(pool,book,position):
    z=np.asarray(pool,dtype=np.float64)[1:].mean((1,2))
    c=np.array([[int(x) for x in word[position*32:(position+1)*32]] for word in book])
    out=(2*c-1)@z/32
    if not np.isfinite(out).all():raise RuntimeError('Nonfinite receiver evidence')
    return out.tolist()

def aggregate(rows,bundle):
    if len(rows)!=40 or {(r['position'],r['condition']) for r in rows}!={(t,c) for t in range(4) for c in CONDITIONS}:
        raise RuntimeError('Incomplete bundle; do not change denominator')
    result={}
    for condition in sorted({r['condition'] for r in rows}):
        selected=[r for r in rows if r['condition']==condition]
        result[condition]=dict(scores=np.mean([r['scores'] for r in selected],axis=0).tolist(),
            eligible=all(r['native']['status']=='VALID' for r in selected))
    return dict(id=bundle['id'],pair=bundle['registered_pair'],conditions=result)

def negative_max(b):
    vals=[]
    for c in RISK_CONDITIONS:
        row=b['conditions'][c]
        # Failed mask abstains, never a successful attribution. Negative evidence
        # uses valid decisions only; all null failures remain separately reported.
        if not row['eligible']:continue
        innocent=[u for u in range(16) if c.startswith(('null','foreign')) or u not in b['pair']]
        vals.extend(row['scores'][u] for u in innocent)
    if not vals:raise RuntimeError('No eligible negative calibration evidence')
    return max(vals)

def calibrate(bundles,expected=64):
    if len(bundles)!=expected or len({b['id'] for b in bundles})!=expected:
        raise RuntimeError('Calibration incomplete')
    values=[negative_max(b) for b in bundles]
    return dict(threshold=max(values),bundle_maxima=values,denominator=expected,
        rule='strict > maximum CAL union-innocent score',sealed=True)

def evaluate(b,threshold):
    calls={c:([u for u,s in enumerate(r['scores']) if s>threshold] if r['eligible'] else [])
           for c,r in b['conditions'].items()}
    pair=set(b['pair']);positive=['registered_mix_png','registered_mix_jpeg85']
    risk=any(set(calls[c])-(pair if c.startswith('registered') else set()) for c in RISK_CONDITIONS)
    return dict(id=b['id'],catch_all_media=all(bool(set(calls[c])&pair) for c in positive),
        catch_png=bool(set(calls[positive[0]])&pair),catch_jpeg=bool(set(calls[positive[1]])&pair),
        false_attribution=risk,returned=calls,
        foreign_native_diagnostic=any(calls[c] for c in ['foreign_0_png','foreign_1_png']))

def risk_upper(k,n):
    return 1. if k==n else float(beta.ppf(.95,k+1,n-k))

def psnr(q,kind):
    return math.inf if q.get(kind+'_psnr_infinite',False) else q[kind+'_psnr_db']

def decide(evaluations,quality,training_times,gates,expected=128):
    if any(len(evaluations[a])!=expected for a in 'ABC'):raise RuntimeError('TEST incomplete')
    ids=[[r['id'] for r in evaluations[a]] for a in 'ABC']
    if ids[0]!=ids[1] or ids[0]!=ids[2] or len(set(ids[0]))!=expected:raise RuntimeError('Unpaired or duplicate TEST')
    b=evaluations['B'];c=evaluations['C']
    win=sum(y['catch_all_media'] and not x['catch_all_media'] for x,y in zip(b,c))
    loss=sum(x['catch_all_media'] and not y['catch_all_media'] for x,y in zip(b,c))
    p=float(binomtest(win,win+loss,.5,alternative='greater').pvalue) if win+loss else 1.
    false={a:sum(x['false_attribution'] for x in evaluations[a]) for a in 'ABC'}
    upper={a:risk_upper(false[a],expected) for a in 'ABC'}
    quality_ok=True
    for a in 'BC':
        for group in ['registered','foreign']:
            q=quality[a][group]
            if q['denominator']!=expected*8:raise RuntimeError('Native quality denominator incomplete')
            quality_ok &= q['bit_accuracy']>=gates['native_bit_accuracy'] and psnr(q,'mean')>=gates['mean_psnr_db'] and q['mean_ssim']>=gates['mean_ssim'] and psnr(q,'min')>=gates['each_native_psnr_db'] and q['valid']==q['denominator']
    for group in ['registered','foreign']:
        quality_ok &= psnr(quality['C'][group],'mean')>=psnr(quality['B'][group],'mean')-gates['psnr_noninferiority_db']
        quality_ok &= quality['C'][group]['mean_ssim']>=quality['B'][group]['mean_ssim']-gates['ssim_noninferiority']
    compute_ok=training_times['C']<=training_times['B']*(1+gates['max_compute_overhead'])
    passed=win-loss>=gates['min_net_wins'] and p<gates['paired_p'] and false['C']<=false['B'] and upper['C']<=gates['risk_upper_95'] and quality_ok and compute_ok
    return dict(status='METHOD_NUMERIC_GATE_PASSED_REVIEW_REQUIRED' if passed else 'C_NOT_SUPPORTED',
        numeric_pass=bool(passed),scientific_pass=None,visual_review='UNRESOLVED',
        wins=win,losses=loss,net_wins=win-loss,paired_p=p,false_attributions=false,
        risk_upper_95=upper,quality_pass=bool(quality_ok),compute_pass=bool(compute_ok),
        denominator=expected,catch_counts={a:sum(x['catch_all_media'] for x in evaluations[a]) for a in 'ABC'},
        evidence_ceiling='One training seed; limited K2/N16/T4/PNG/JPEG85; no generative or novelty PASS')
