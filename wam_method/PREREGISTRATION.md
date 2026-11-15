# WAM writer A/B/C — frozen single-seed comparison

Registered before model execution, 2026-11-15. Execution ID:
`wam-writer-abc-20261115-attempt01`. This is a new, separately budgeted
method comparison authorized after the WAM visual/pair review. It does not
change, reopen, or retrospectively pass qualification attempt02.

## Question and evidence ceiling

Does a task-level innocent-tail ranking objective improve catch-one attribution
over ordinary mixture training at matched payload, channel, training calls,
quality, and independently calibrated rejection?

The existing WAM qualification establishes a usable image writer/extractor and
continuous evidence interface. It does not establish a CollTM gain. Visible
color-edge changes in the prior native-resolution review prohibit describing the
writer as visually imperceptible. This experiment neither establishes a
generative/model-distributed watermark nor validates VAE/MOLM transfer.

## One candidate and two controls

* **A**: official WAM writer, evaluation only.
* **B**: writer-only AdamW training on native copies and genuine equal-weight RGB
  mixtures, soft-target bit loss plus native/mask/quality losses.
* **C**: the identical B objective plus one normalized task-level ranking loss.

Official source: facebookresearch/watermark-anything commit
`2c08af04d037d5667c02f6ddebbda9ff04581c3e`, `wam_mit.pth`, `vae_small` writer,
`sam_base` extractor, 32 bits, JND `jnd_1_3_blue`, scaling_w=2, scaling_i=1.
The frozen extractor and JND stay in eval mode with no parameter gradients.
Writers use float32/eval mode; optimizer updates their parameters only. No SD
generator or prompt is used.

At each update, the same four images, 16-user 128-bit codebook and two colluders
are presented to B/C. Each image carries its fixed contiguous 32-bit slice.
Let z concatenate the four mixed-image global mean bit-logit vectors. Define
S_u = mean((2c_u-1) z), q = ||z||_2/sqrt(128) + 1e-6, and
L_tail = softplus((max_innocent S_u - max_colluder S_u)/q).
Coalition membership enters TRAIN supervision and offline evaluation only;
the blind receiver never uses it to select candidates or set scores.

Per-work base loss is native BCE (two copies averaged) + mixture BCE against
the bitwise mean of the two payloads + mask BCE (three views averaged) +
100 × image MSE (two copies averaged). Base loss is averaged over four works;
C adds L_tail with coefficient 1. There is no additional common-bit loss.

Both arms make the same two forward passes per work, including the same leaf
gradient calculation. The second pass propagates an exact streamed VJP through
the four work slices; B's extra rank gradient is multiplied by zero. This controls
forward/backward calls rather than granting C additional compute. Alternating
B/C order reduces order-dependent runtime bias. Wall time is reported separately.

AdamW: lr=5e-5, betas=(.9,.999), eps=1e-8, weight_decay=.01;
gradient norm clip=1. Exactly 500 complete bundle updates per arm, one seed
2026110902, no scheduler, no early-checkpoint selection, final checkpoint only.
Nonfinite loss/gradient or recomputation mismatch stops the package.

Native output is clamped and rounded as saved PNG; RGB mixtures use rint.
Even-numbered updates use mixture PNG, odd updates actual Pillow JPEG quality85,
subsampling2. Forward JPEG is the real encode/decode operation; backpropagation
uses an identity straight-through surrogate. B/C share this surrogate. TEST
uses actual saved-and-reopened PNG/JPEG bytes, never approximate JPEG.

## Data and one-time sampling

Official COCO `image_info_test2017` metadata supplies identities, dimensions and
licenses. Candidate images require license ID4 (CC BY2.0 metadata) and minimum
dimension256. From 6,405 eligible records, 1,024 were drawn once under the frozen
seed/domain. The exact metadata, image URLs, codebooks, keys, pairs and all
updates are fixed in `manifest.json`; `freeze_wam_method.py` refuses overwrite.

* TRAIN: 256 images; 500 bundles sample four distinct TRAIN images each.
* CAL: 256 different images, 64 nonoverlapping four-image bundles.
* TEST: 512 further images, 128 nonoverlapping four-image bundles.

RGB conversion, bicubic shorter-edge resize to512, center crop512 are fixed.
Local source checks decode every image and reject exact decoded duplicates,
missing/invalid source data or dimension mismatch without replacing samples.
There is no result-dependent image or codebook selection. These are our watermark
protocol splits, not evaluation against COCO challenge labels.

Available Model-WM usage records show CIFAR100 QIM sources, the prior AquaLoRA
run and 16 WAM authored DEV images, without COCO test2017 use. Actual checkpoint
training overlap and near-duplicate independence remain UNVERIFIED. A metadata
license label is not an independent re-verification of the original photograph's
license. Source links and metadata are retained. This qualification limitation
is not erased by disjoint project IDs.

## Receiver, CAL, TEST

Each receiver score is the ordinary signed global-mean raw-logit score,
averaged across all four works. There is no phi correction, HMAC sampling,
truth-selected candidate, or score rescaling at evaluation. All four official
native detector masks must be valid for a bundle-condition to return identities;
otherwise it abstains and remains in the denominator. Hard native decoding uses
the official masked pooling and .5 bit threshold; raw sign common-bit diagnostics
are reported separately.

Ten conditions per image: unmarked PNG/JPEG85; two registered native copies;
two foreign native copies; registered equal RGB mixture PNG/JPEG85; foreign equal
RGB mixture PNG/JPEG85. Foreign payloads are independently drawn codewords, not
registered users, even when some bits agree.

For each CAL bundle, take the maximum score of any innocent registered user over
six conditions: unmarked PNG/JPEG, foreign mixture PNG/JPEG, and registered
mixture PNG/JPEG (exclude the two actual colluders only for this last pair).
Separately for A/B/C, seal the maximum of 64 bundle maxima as the threshold.
Return every eligible identity with score strictly greater than the threshold.
All three CAL thresholds must be sealed before any TEST extraction starts.
No eligible negative calibration evidence is a failure, not a synthetic threshold.

The primary paired outcome is catch-one on **both** registered PNG and JPEG in
the same TEST bundle. A false-attribution event is the union across all six
conditions and all innocent identities in that bundle. Foreign native copies
are additional diagnostics outside this six-condition primary family.
Preserve all per-work pools, masks, native bits, scores, common-bit diagnostics,
quality, returned identities, failures, and paired A/B/C bundle results.

## Pass and stopping conditions

All conditions must hold; otherwise C is not supported in this frozen setting:

1. Complete both 500-update arms and every CAL/TEST operation, within budget.
2. C minus B joint-catch net wins at least13/128; one-sided exact paired
   binomial/McNemar p<.05. A is a descriptive baseline, not another selected test.
3. C's union false-attribution count is no greater than B's; its one-sided
   Clopper–Pearson95 upper bound is at most5%. Even 0/128 gives about2.31%, so
   this cannot establish a 1% or rarer-risk claim.
4. For both B/C, separately registered/foreign native groups: bit accuracy>=.95,
   mean PSNR>=23dB, every native PSNR>=20dB, mean SSIM>=.75, no invalid masks.
   Each TEST group has1,024 native images. C's mean PSNR may trail B by at most
   .5dB and SSIM by at most.005.
5. C's measured training compute-section wall time is at most1.05×B's;
   both pay equal physical forward/backward calls. Drive publication overhead is
   charged globally but excluded equally from per-arm compute-section timing.

Numerical success remains `METHOD_NUMERIC_GATE_PASSED_REVIEW_REQUIRED`:
native-resolution visual review and scientific adjudication are separate.
One paired training seed cannot establish cross-seed robustness or novelty.
Diagnostic risk curves use a CAL-only fixed quantile grid
[0,.5,.75,.9,.95,1]; they never change the predeclared TEST operating point.

## Fixed execution budget and physical work

One new package: 21,600 allocated-GPU seconds (6h), 28,800 wall seconds (8h),
21,600 CPU core-seconds, 20GiB results, 32GiB RAM, 24GiB device memory.
GPU allocation is conservatively charged for the whole entry, including setup.
Preparation <=1,800s; each training arm<=3,600s; evaluation<=10,800s;
60s is reserved for cleanup and final sealing. These are simultaneous limits.

Preparation allowance exceeds three times the observed approximately495-second
completed cold preparation, rounded with headroom. The previous 16-work run
observed about3.59s/work after removing generation, suggesting roughly2.30h for
2,304 evaluation work-arm instances. These are planning observations, not a
guarantee. Real backward cost has not been measured. After the first16 formal
updates of **both** arms, their observed maximum step costs ×1.25 project the
remaining484; an infeasible projection stops. Those16 count inside500; there is
no extra warmup, pilot, shorter substitute, new seed, or second training trial.

Physical totals: 0 generation; 500+500 updates; 16,000 training writes and24,000
training extracts; 9,216 evaluation writes and23,040 evaluation extracts.
Every physical call is reserved before execution, including failed calls.

Historical AquaLoRA and WAM costs are inherited from the latest sealed attempt02
summary, floored at the pre-registration receipt. Measured CPU remains a lower
bound where old interruption tails are unknown; conservative debits remain
separate. The new budget does not refund or retroactively pass old budgets.

## Delivery and failure behavior

The user selects GPU and clicks Run all. First cell only mounts Drive; the
second code cell unpacks an embedded frozen package and starts this execution.
No code upload, mutable branch checkout, weight upload, or agent-started GPU.
Live IPC, environment, source images, downloaded weights and caches stay in
`/content`. Existing compatible environments and paid caches are checked and
reused. A replaced runtime must validate/install its environment again.

Only a transient image transport failure gets one additional request to the
**same URL**, within preparation budget. Both request receipts and partial-byte
costs are retained; 404/source/format errors stop immediately. This is a bounded
transport recovery, not an extra sample, model call or experimental retry.
The engineering preflight encountered three TLS EOFs, then obtained those same
three images; both source-check receipts are retained in the delivery audit.

Drive stores sealed observations, logs, results and newly trained final B/C
writers (research outputs, not a cache of downloaded pretrained weights).
Old attempts are read only. A repeated cell reads the saved result; an unsealed
interrupted package stays stopped and reserves its full budget, without rerun.
Final summary writes are covered by the60-second conservative debit; measured
totals explicitly end at quiescent artifact archival. Missing or corrupt
historical receipts stop execution. No automatic resume of consumed model work.

CPU and random-state interface tests do not prove real-weight GPU feasibility.
No new Gemini review was performed for this delivery. No numerical, visual,
CAL/TEST, anti-collusion, generative-transfer or innovation PASS is claimed before
the authorized user-run package produces its evidence.
