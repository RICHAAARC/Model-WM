# SD2.1 denoiser / fixed LaWa decoder — frozen 25 November 2026

This is cross-denoiser replication, not a new watermark method. The existing SD1.4 paper and its 80-work result stand independently. No second-model outcome is known at registration.

## Source and execution

Use the HF-documented community mirror `sd2-community/stable-diffusion-2-1-base` revision `4e63672c03103b6c636b8fb4119ba982469b2955`. This is not a currently maintained Stability AI distribution. UNet/text encoder/tokenizer are from this fixed source. The clean VAE stays CompVis SD1.4 revision `133a221b8aa7292a167afc5127cb63fb5005638b`; LaWa source and checkpoint remain the prior versions. No SD2 VAE checkpoint is used. The construction is **SD2.1 denoiser + fixed KL-f8/LaWa decoder**, not the untouched SD2.1 pipeline. Shared latent convention is supported by source/configuration evidence; image quality and watermark compatibility require the actual DEV run.

512×512, DDIM 50, CFG 7.5, eta 0, epsilon prediction; float32 throughout. Fixed staged transfers: pipeline on GPU for denoising, writer on CPU; then text encoder/UNet on CPU, fixed clean VAE and writer on GPU for decoding/writing. No precision or sampler fallback after an OOM. Divide terminal latent by 0.18215 once; writer post_quant_conv once. Codes are ±1 at the writer. Continuous reader output has 48 entries. The exact scale-transfer and uniform-gain implementations are reused unchanged.

## Data and work limits

112 new manually authored prompts are assigned once by master seed 2026112501 to **8 DEV / 24 CAL / 80 TEST**, with generation seeds, three registered identities, three foreign identities, arm order and DEV action order fixed in allocation.json. Historical manifest union includes 504 unique prompts and all reserved splits. Exact prompt/seed reuse is rejected; nearest lexical neighbors are recorded. Semantic overlap and unknown checkpoint training data are not certified absent. The accepted unseen-content assumption remains explicit.

Same 32 codewords of 48 bits and same 16-person registered roster as SD1.4. No code redraw. All missing, failed and zero-difference works keep their denominator. No seed swap, extension, post-hoc pooling or repeat after a started physical call.

Fixed family totals: **112 generate / 1488 write / 1248 mix / 5224 read**.
This DEV notebook performs only **8 generate / 240 write / 24 read / 0 mix**.
It does not load or execute the CAL/TEST work rows.

## DEV selection and interface

First two DEV works count within the eight. For those works, six single copies from fixed coarse and six from ordinary gain=1 are each read exactly once, totaling 24 continuous reader checks. Only shape/finiteness/call-chain checks are gates; payload scores do not select gain or scale. Save both works at original resolution, interface metrics and all 24 readouts for review. Invalid tensor/RGB/reader output stops immediately. Human/AI image review after DEV checks subject and layout, gross noise/color fringes/ghosting and pseudo-text changes before authorizing CAL. Engineering screen is not a human blind quality comparison.

Coarse transfers 10% of the smaller of coarse/fine branch energy toward coarse, leaving middle branches unchanged. Ordinary uniformly scales all six branches by 1.000, 1.025, 1.050, 1.075. Every DEV work receives all five actions and six identities. Choose ordinary gain minimizing absolute mean log(ordinary/coarse) structural loss, breaking ties toward lower gain. Structural loss is the unchanged RGB 128×128 BOX 1−SSIM proxy, averaged across six copies. Only this quality criterion selects gain. Require geometric ratio in [0.98,1.02]; otherwise stop, no wider grid. All eight complete plus numeric interface check yields DEV_MATCHING_GATE_MET; visual review remains PENDING and scientific_pass=null. No attribution efficacy claim follows from DEV.

## Later stages (not reachable from this notebook)

After DEV receipt/image audit, independent 24 CAL uses the fixed arms and the same full query family. Per-work maximum over all innocent registered IDs and all queries, then maximum over 24 works, defines the shared threshold, with strict score > tau. Seal and read back CAL threshold before first TEST operation. No SD1.4 numeric threshold reuse. This calibration has 1/25 rank resolution under exchangeability, not certified low FWER.

80 TEST: registered and foreign singles; equal-two, equal-three and (.5,.25,.25) mixtures; clean controls; PNG and JPEG85 4:2:0. 50 physical reads/work, sharing clean reads. Only registered equal-three PNG member count per work is primary. One-sided exact paired sign test p≤.05 and positive total member gain are required. Safe capture, JPEG and other mixtures are separate endpoints, not extra independent works.

Require no coarse-new innocent identity over the full query family, structural-loss 90% t interval entirely within [log(.98),log(1.02)], completed fixed operations and acceptable image continuity. Shared errors remain errors. Any failed gate stops a positive replication claim; retain negative and incomplete outcomes without extra samples or alternative models. Existing SD1.4 results cannot rescue a failed SD2 primary endpoint through pooling.

## Accounting and replay

No arbitrary preparation/model total-time hard caps. Limits are scientific operation counts; estimates are not deadlines. Estimated DEV preparation 5–20 minutes and model 5–15 minutes, hardware/network dependent. Full-family model estimate 70–120 minutes. Resources are metered for preparation, model, settlement and full entry; inherit the final SD1.4 cumulative actual and conservative debit ledgers separately. Unknown historical CPU remains inexact. Download accounting is new retained payload high-water, not network packets; reused validated local files are not charged again. Weights/packages only /content; Drive stores receipts/results.

Repeated Run all reads saved receipts, including failures; it cannot restart model calls. An interrupted entry exposes an observed lower bound and unknown tail, never a fake final budget. Runtime replacement loses /content environment/cache; reuse requires a current environment probe and valid local cache receipt, not a historical READY flag. No automatic recovery run is authorized by this notebook.
