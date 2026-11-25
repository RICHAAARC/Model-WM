"""Quality-only DEV choice and predeclared four-arm adjudication; no models."""
import math
from statistics import mean,stdev
from scipy.stats import binomtest,t,beta
GAINS=(1.020,1.025,1.030,1.035,1.040,1.045)
ARMS=('coarse','low','near','high')
N_DEV,N_CAL,N_TEST=8,24,160
DEV_LIMITS=dict(generate=8,write=336,read=0)
PRIMARY='reg/equal3/png'
BANDS={'low':(.985,.995,.990),'near':(.995,1.005,1.),'high':(1.005,1.020,1.010)}

def choose(rows):
    if len(rows)!=N_DEV or len({r['work'] for r in rows})!=N_DEV:raise ValueError('Need all eight DEV works')
    keys={'coarse'}|{str(g) for g in GAINS}
    for row in rows:
        if set(row['loss'])!=keys or any(not math.isfinite(v) or v<=0 for v in row['loss'].values()):raise ValueError('Incomplete or invalid DEV quality')
    ratios={str(g):math.exp(mean(math.log(r['loss'][str(g)]/r['loss']['coarse']) for r in rows)) for g in GAINS}
    monotone=all(ratios[str(a)]<ratios[str(b)] for a,b in zip(GAINS,GAINS[1:]))
    selected={}
    for arm,(lo,hi,target) in BANDS.items():
        eligible=[g for g in GAINS if lo<=ratios[str(g)]<=hi]
        if eligible:selected[arm]=min(eligible,key=lambda g:(abs(math.log(ratios[str(g)]/target)),g))
    ok=monotone and set(selected)=={'low','near','high'} and selected['low']<selected['near']<selected['high']
    return dict(status='DEV_BRACKET_GATE_MET' if ok else 'DEV_BRACKET_STOP',selected_gains=selected,
                geometric_loss_ratios=ratios,monotone=monotone,bracket_gate_passed=ok,
                selection_input='DEV quality only; zero reader calls',cal_test_authorized=False,
                visual_review='PENDING',scientific_pass=None)

def quality_interval(values):
    if len(values)!=N_TEST or any(v<=0 or not math.isfinite(v) for v in values):raise ValueError('Need 160 valid ratios')
    logs=list(map(math.log,values));m=mean(logs);half=float(t.ppf(.95,N_TEST-1))*stdev(logs)/math.sqrt(N_TEST)
    return [math.exp(m-half),math.exp(m+half)]

def adjudicate(d,safe_d,coarse_new_false,high_new_false,quality,complete=True):
    """Only one confirmatory contrast: coarse minus high on equal-three PNG."""
    if len(d)!=N_TEST or len(safe_d)!=N_TEST or any(x not in range(-3,4) for x in d) or any(x not in (-1,0,1) for x in safe_d):raise ValueError('Invalid work differences')
    bounds={a:quality_interval(quality[a]) for a in ('low','near','high')}
    valid=complete and bounds['low'][1]<=1 and .995<=bounds['near'][0]<=bounds['near'][1]<=1.005 and 1<=bounds['high'][0]<=bounds['high'][1]<=1.02
    w,l=sum(x>0 for x in d),sum(x<0 for x in d)
    p=float(binomtest(w,w+l,.5,alternative='greater').pvalue) if w+l else 1.
    if not valid:status='BRACKET_NOT_ESTABLISHED'
    elif p<=.05 and sum(d)>0 and not coarse_new_false:status='COARSE_GAIN_PERSISTS_PENDING_VISUAL'
    elif sum(d)<=0 and sum(safe_d)<=0 and not high_new_false:status='ORDINARY_CATCHUP_IN_THIS_SAMPLE'
    else:status='INCONCLUSIVE'
    upper=float(beta.ppf(.95,w+1,N_TEST-w)) if w<N_TEST else 1.
    return dict(status=status,scientific_pass=None,primary=PRIMARY,denominator=N_TEST,wins=w,losses=l,ties=N_TEST-w-l,
                net_members=sum(d),mean_members=mean(d),net_safe=sum(safe_d),one_sided_exact_sign_p=p,
                quality_ratio_90ci=bounds,bracket_valid=valid,no_coarse_new_false=not coarse_new_false,
                no_high_new_false=not high_new_false,coarse_win_probability_upper95=upper,
                zero_win_positive_gain_upper95=3*upper if w==0 else None,
                population_equivalence_claim=False,visual_review='PENDING')
