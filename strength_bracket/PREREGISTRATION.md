# Residual-strength bracket diagnostic — frozen 25 November 2026

## Question and immutable scope

Does the fixed 10% coarse-branch transfer retain an identity-recovery advantage when ordinary uniform strengthening incurs slightly greater structural loss? This supplements, but does not rejudge, the two earlier 80-work tests. It is not a third generator, new carrier, training experiment, attack search or perceptual-equivalence study. Use the original SD1.4/LaWa graph, 48-bit codebook, 16-user registry, DDIM50 CFG7.5 eta0, float32, 512×512, same per-work latent and identity triples across arms. Preserve genuine RGB writes, mean-then-JPEG85 (4:2:0), reader and information access. No TEST-based choice of gain or threshold.

## Once-only allocation and DEV

Master seed 2026112502 freezes 192 new works, generation seeds, identities and arm orders: **8 DEV / 24 CAL / 160 TEST**. Historical/reserved manifests (including both prior 80-work tests) are checked for exact normalized prompt and seed collisions. Repeated topics and the 12 neutral setting templates are permitted; independent works do not imply disjoint semantic categories or known pretraining exposure. All missing or failed works retain denominators. No redraw, seed swap, sample extension or model-call rerun.

Eight DEV works receive the fixed coarse action and six global uniform gains **1.020, 1.025, 1.030, 1.035, 1.040, 1.045** in frozen order, each for three registered and three foreign copies. Exactly **8 generate / 336 write / 0 mix / 0 read**. First two works are included and retained at original resolution for interface/image review. There is no efficacy extraction or per-work gain choice.

Use the unchanged mean six-copy RGB 128×128 BOX 1−SSIM quality loss. For each gain compute the geometric mean ordinary/coarse loss ratio across all eight works. Require strictly increasing geometric ratios over the frozen grid. Choose three distinct ordered gains with smallest absolute log distance to their target, ties toward lower gain, within these fixed bands:

| Role | Target ratio | Eligible DEV ratio |
|---|---|---|
| low | .990 | [.985,.995] |
| near | 1.000 | [.995,1.005] |
| high | 1.010 | [1.005,1.020] |

If any band is empty, choices are not strictly ordered, completeness/validity fails, or the monotonicity gate fails: STOP, retain all outcomes, do not widen the grid. Otherwise save `DEV_BRACKET_GATE_MET`, gains and all losses, with visual review pending and scientific_pass=null. The entry cannot enter CAL/TEST. After receipt and image review, a separate formal entry may use the sealed gain selection; no DEV repetition.

## Independent calibration and test

Four arms: coarse, low, near, high. For each CAL work take the maximum score over all four arms, all 26 query conditions and every innocent registered ID; the maximum of the 24 work maxima is one shared threshold. Strict score > threshold returns every qualifying ID. Seal and read back the threshold and completed CAL records before reserving any TEST operation. The 26 conditions per arm are six singles, six mixtures (registered/foreign × equal2/equal3/unequal3), one clean, each in PNG/JPEG85. Clean reads are shared across arms: **98 physical reads/work**. No arm-specific tuning, threshold update or ranking of raw uncalibrated scores.

Each new work supplies actual registered and foreign RGB copies for all four arms. Complete formal totals: **184 generate / 4416 write / 4416 mix / 18032 read**. Entire family including DEV: **192 / 4752 / 4416 / 18032**. Negative, missing and tied works stay visible. Training updates are zero. Old holdouts stay sealed.

## One primary contrast; three outcomes

Only confirmatory efficacy contrast: registered equal-three PNG, `d_w = true-members(coarse) − true-members(high)`, one difference per TEST work. One-sided exact sign test on nonzero work differences; all 160 remain in effect denominators. Require p≤.05 and positive sum(d). Low/near comparisons, safe capture, JPEG and other mixtures are secondary and cannot rescue a primary failure. Do not pool earlier studies or multiple codecs to increase n.

Quality establishment uses 90% two-sided Student-t intervals for the mean log ratio, exponentiated. Across 160 works require low/coarse upper≤1; near/coarse interval wholly within [.995,1.005]; high/coarse lower≥1 and upper≤1.02. This verifies the high ordinary control incurred greater structural-proxy loss in this sample. It does not prove equal perceptual cost. Retain all ratios and original-resolution artifacts. A failed bracket or incomplete family is `BRACKET_NOT_ESTABLISHED`, not evidence for either mechanism.

With a valid bracket and acceptable subject/quality inspection:

1. **Coarse gain persists:** primary p≤.05, sum(d)>0, and no coarse-new innocent ID relative to high anywhere in the full query family. Supports a scale-allocation benefit that the tested slightly higher uniform strength did not eliminate. It is not complete causal isolation or a general robustness claim.
2. **Ordinary catches up in this sample:** high returns at least as many total primary true members and at least as many primary safe captures, and no high-new innocent ID relative to coarse over the full family. This weakens the stronger scale-specific interpretation; it does not erase prior frozen results or establish population equivalence.
3. **Inconclusive:** all other complete valid outcomes. No additional samples, gains or reruns.

Report absolute errors and pairwise new/common errors for every arm. Absence of newly introduced errors is not zero risk; 24-work calibration has rank resolution 1/25 under exchangeability. If **zero** TEST works favor coarse over high, additionally report the exact one-sided 95% upper bound on coarse-win probability, `1−.05^(1/160)=.0185491341`. Because positive member difference is at most three, the corresponding conservative upper bound on mean positive advantage is .0556474022 members/work, below the smaller historical point effect .075. This is a conditional bound under independent/exchangeable work sampling, not a general equivalence theorem or a claim that old .075 was known without uncertainty. If any coarse-win occurs, retain the general probability bound but do not use this zero-win conclusion.

## Size justification and stopping

160 TEST works provide exact sign-test power about 99.4% for win/loss probabilities (.075,0), 91.6% for (.0875,.0125), and 83.3% for (.10,.025). At (.075,.025), power is 56.9%; smaller effects may remain inconclusive. These are design scenarios, not outcome estimates. At n=80 the corresponding powers are 72.5%,57.2%,48.4%,26.2%. Thus 160 targets the previously observed sparse directional effect with allowance for some reversals, without claiming power for all joint quality/risk gates.

Fixed scientific work counts, no arbitrary preparation/model wall-time hard stop. Expected DEV preparation 5–20 min and model 5–15 min; formal model roughly 150–270 min based on prior two-arm measured throughput with four-arm/184-work scaling, not a deadline. Actual preparation, model, settlement, CPU, allocated GPU clock, storage and downloads are metered separately. Current-runtime validated environment/assets are reused; all weights/packages remain in /content. Drive stores records/results. Runtime replacement may require download again. Preserve all earlier actual and conservative debit ledgers from the completed SD2.1 comparison. New A costs and B CPU analysis are separate; no cross-line scientific-data reuse. Failures retain observations and unknown tails, and reopening reads saved results without repeating physical calls.
