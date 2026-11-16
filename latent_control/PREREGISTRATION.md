# Shared VAE response control — frozen attempt01

Frozen before real weights, generation, CAL or TEST. User adopted the single route on 2026-11-16.
Run: `latent-control-20261116-attempt01`. Master seed **2026111601**. Sampling occurred exactly once;
`manifest.json`, `frozen.npz`, and `source_usage_history.json` are the executable frozen specification.
This is a new bounded mechanism probe, not a retry of any QIM/AquaLoRA/WAM attempt. All their failures
and cumulative charges remain closed and inherited. The prior WAM C remains `C_NOT_SUPPORTED`.

## Hypothesis and scope

Trusted service, same prompt/seed/work, two recipients mixing their actual uint8 RGB copies and then
JPEG85. N=16 registered identities, K=2, T=1, 512x512. The 64 orthogonal identities carry at most
**6 identity bits**, not 64 independent payload bits or the old 128-bit cross-work protocol.
No parameter merging, crop, mosaic, denoising/regeneration, or white-box forgery claim.
Work ID may be trusted. Receiver sees only image, fixed reader/codebook and sealed threshold.

New authored prompts are uniformly sampled without replacement from the retained vocabulary;
136 distinct prompts, seeds and work IDs: TRAIN=0, DEV=8, CAL=64, TEST=64. DEV never selects a setting.
The previous inspected TEST data are used-history, not new independent evidence. Semantic overlap
with pretraining is unverified; the user's unseen-content assumption remains explicitly labelled.

## Frozen generator, writer and receiver

Reuse SD1.5 revision `451f4fe16113bff5a5d2269ed5ad43b0592e9a14`, 25 DPM multistep steps, CFG7.5,
float32, one shared denoising trajectory per work. Freeze all network weights. Divide pipeline latent
by its actual `vae.config.scaling_factor`, then add the control **before** VAE post_quant_conv/decoder.
No RGB residual bypass and no encoding of a completed image. No new trained extractor is required.

P is fixed linear normalized-luminance 8x8 orthogonal DCT, frequencies (1,2),(2,1),(2,2),(1,3),
(3,1),(2,3),(3,2),(3,3), with a once-frozen permutation/sign spreading into 64 disjoint normalized
groups. PP^T=I. Fixed signed/permuted Hadamard columns form unit codes. IDs 0..15 are registered;
foreign DEV/CAL/TEST use disjoint pools 16..31/32..47/48..63. Every work has two registered and
two foreign copies, selected before execution. No resampling based on condition number or image.
The published key is a reproducibility key for this restricted experiment, not a production secret.
Its actual reader key material is the explicit permutation/sign arrays in `frozen.npz`;
`manifest.key` is their run-local identifier, not a seed used to resample them at execution time.
Foreign means an unregistered identity under this same key/code system, not an arbitrary other issuer.

rho=3/255 RGB RMS; a=rho*sqrt(3*512*512)/2. r(z)=P D(z), J=dr/dz,
B=J^T(JJ^T+lambda I)^(-1), lambda=1e-4 trace(JJ^T)/64.
C computes u0=-B(z)r(z), zc=z+u0, then recomputes Bc exactly once;
u_j=u0+Bc(a*c_j-r(zc)). Rank failure at sigma_min<=1e-6*sigma_max is retained, never repaired by
another key, layer, frequency, seed, strength or extra iteration. Matrix solve uses float64;
VAE and controls remain float32. Decoder D is normalized floating RGB before clamp/quantization;
all evaluated endpoints are actual clamp-and-round uint8 images. That discrepancy remains in errors.

B is ordinary per-copy latent informed optimization with the same target, reader, generator and
quality cap: ||P D(z+u)/a-c_j||² + relu(MSE(D(z+u),I0)/rho²-1)²,
where I0 is the saved clean uint8 RGB converted back to [0,1].
L-BFGS: lr=1, history=10, strong Wolfe, max_iter=32, max_eval=32, tolerance_grad=1e-7,
tolerance_change=1e-9. An additional closure guard reserves before physical execution and prevents
even a line-search overrun beyond 32 evaluated gradients/copy. Best quality-feasible **already
evaluated** point is retained; no extra decode, JPEG-based selection or retry. Every evaluated
objective and quality is retained. The guard is a normal exhausted budget, not an extra run.

Arm order alternates per frozen work. Both arms share the same clean generation; all candidate
precomputation and both arms' quality/IO/failed work count. Other common costs are divided equally.
No gradient updates to model parameters. TF32/benchmark disabled; package versions locked.

Receiver q=P(image)/a, s_j=c_j^T q, strict s_j>tau; no forced top1, abs or norm normalization.
For an orthogonal-code weighted mixture, q=sum beta_i*c_i+error. If all registered directional errors
are <=delta and delta<tau<max(beta)-delta, catch-one/no-innocent follows conditionally. JPEG/rounding
errors are measured, not assumed small. This is not a global FWER proof or proof of novelty.

## Physical operations and channel

Every work: two registered and two foreign originals per arm; registered/foreign pair mixtures at
weights .25/.5/.75; clean image; PNG and actual JPEG85, subsampling2, optimize=False. Mixture is
numpy.rint of weighted uint8 RGB. JPEG bytes are stored and decoded exactly once for scoring.
22 image decisions/arm/work. A missing native also leaves its mixtures missing; missingness is retained.

Hard physical ceilings: 136 generations, 3400 denoiser calls (CFG sample-equivalent work also applies),
544 writes/arm, C816 decode forwards, B17408 objective/forward/reverse calls, C17408 reverse calls,
5984 evaluation reads. DEV has only 16 additional decode forwards for a fixed first-coordinate
central finite difference (h=.01, atol=.0005, rtol=.05); these are charged. Finite-difference mismatch
stops as an interface failure, never tunes h/tolerances. No new forward after a physical limit.
STARTED consumes the call even if it fails. No failed/missing item is removed from the denominator.
Model operations synchronize before COMPLETED, so an asynchronous CUDA failure cannot be called completed.

## CAL sealing, TEST and frozen decisions

Both arms complete all 64 CAL works. Per-work risk family includes null, foreign native, foreign mix,
registered native nonowner and registered mix nonmember, across both codecs and all mix weights.
tau is the maximum of all 64 work-level union-innocent maxima. Missing/nonfinite risk evidence stops
before TEST. Both CAL thresholds must be sealed to Drive and read back identically before the first
TEST generation. Truth is used for CAL risk and later adjudication only, never in receiver calls.

TEST has exactly64 paired works. Work success requires all six registered mixture views catch-one
and all four native copies present and quality-eligible. This is the adopted conservative eligibility
rule, including foreign-copy quality. Native quality: MSE<=rho² (PSNR>=38.5883785dB) and SSIM>=.98.
SSIM implementation: skimage 0.24, channel_axis=2, data_range=255, default uniform 7x7 window.
Rounding 38.59 in the proposal does not replace the exact rho² rule.

Numeric gates: C>=48/64; paired net C-B>=8/64; exact one-sided discordant binomial p<=.05;
C union-risk errors=0/64 (95% one-sided upper bound 4.57297%, not zero risk); fixed 256 TEST copies
per arm available for quality means; C mean PSNR no worse by>.1dB and SSIM by>.001;
C complete wall and CPU cost<=B, including the same split of all common overhead.
An unchanged image has explicit infinite PSNR; it cannot silently improve or disappear from a mean.
All tests are fixed; no alternate TEST threshold, objective, seed, key or attack is selected.

Numeric success remains `NUMERIC_GATES_PASSED_VISUAL_PENDING`. Original-resolution visual review and
the saved response/endpoint/codec-error evidence require review before any final mechanism claim.
No automatic scientific/innovation PASS. A complete failed numeric gate gives `C_NOT_SUPPORTED`.
Source/interface/resource interruption is incomplete, not a scientific negative, and cannot auto-resume.

## Independent budgets and failure retention

Preparation wall<=1800s. Model loading through generation/writing/evaluation/sealing wall and GPU
allocation proxy<=21600s, including60s sealing reserve. Total CPU<=43200 core-seconds, RAM32GiB,
VRAM16GiB, results10GiB. Preparation does not consume the model time envelope, but remains in the
complete allocation/cost history. Download is durable new-payload high-water, not network packets.
Paid caches are not charged again. Historical exact CPU is unavailable; cumulative CPU stays a lower
bound while the historical conservative debit is separately inherited and never refunded.

After8 DEV, project remaining128 works using 1.5 times the maximum complete DEV work (both sequential
arms, generation and archival) plus all costs already incurred and reserve. Stop before CAL if any
projected time/CPU/result budget exceeds its cap. This is conservative screening, not a timing guarantee.
Runtime resource checks continue; no extra warmup, tuning, shortened TEST or additional attempt.

Local /content holds environment, all pretrained weights/cache, and dynamic IPC. Drive holds only
immutable observation receipts, source/protocol snapshots, closed results and failure records. Prior
Drive caches can be read/copied to /content; no new pretrained weights are saved in Drive. Current
interpreter and exact packages are probed; an old READY receipt never proves a runtime survived.

Read the latest sealed `wam-writer-abc-20261115-attempt01/summary.json` at startup and take monotone
max with the included historical floor. Preserve its nested previous costs and all prior directories.
Current inherited wall observation17644.495509s, conservative wall debit17804.495509s;
CPU lower bound15789.114133 core-s, conservative CPU debit16106.299188 core-s.
Original historical envelopes remain closed; this new explicit envelope is separately labelled.

Reserve the attempt before installation/model work. Reopening a sealed attempt reads its summary;
an unsealed reservation is read-only and retains the full attempt allowance as a conservative unknown-tail
debit. No automatic retry, resume, deletion or cleanup of old attempts. Full exception stacks retained.

## Engineering evidence ceiling

CPU tests use mathematical fixtures and randomly initialized official component architectures only.
They establish executable interfaces, counting, metadata parsing and control flow, not real-weight
quality, feasibility, budget completion or scientific success. No real weights or GPU were used to
prepare this package. The GPU run is initiated only by the user in Colab.
