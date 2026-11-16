# Delivery evidence — 2026-11-16

Implemented the user-adopted generation-internal shared-response mechanism and one ordinary latent
optimizer control. A single frozen sampling produced `manifest.json` and `frozen.npz`; the freezer
now only validates existing artifacts. No experimental parameters, seeds or identities were selected
using generated outputs. No pretrained model weights were downloaded or loaded for delivery.

## CPU verification

- 40 CPU regressions: linear/orthonormal readout, frozen split and codebook, quantitative codec and
  quality rules, L-BFGS hidden line-search cap, per-work denoiser cap, retained partial failures,
  actual shared/ordinary algorithms on synthetic carriers, and full synthetic DEV/CAL/TEST worker.
- CAL seal/readback before TEST, and CAL failure blocking TEST; foreign native in the formal risk family.
- Concurrent atomic IPC; exclusive operation reservations; real failed/cancelled child process reaping;
  no budget refunds; separate preparation/model envelopes; read-only reopening; unknown-tail reserves;
  self-containing final result-volume settlement.
- Actual pinned SD1.5 component index and all 10 small metadata/config/tokenizer assets parsed with
  Python3.10.14, torch2.5.1 CPU, diffusers0.31.0 and transformers4.44.2. Random initialization of the
  **full official VAE architecture**, 512px forward and latent VJP, frozen parameter gradients checked.
- Full official pipeline classes saved/loaded with synthetic small tensor state and the real component
  index, tokenizer and processor; DPM scheduler,25 denoiser calls, scaling factor and latent gradients.
- Three official checkpoint HEAD availability probes only; no payload GET. Availability is not weight
  correctness or real model performance.
- Notebook schema, first-cell mount-only, exact embedded sources/frozen arrays and real host bootstrap
  tested against a read-only saved receipt. No installation or model subprocess was started by this test.

Records are in `review/cpu_tests.log`, `review/official_cpu.json`, `review/official_weight_head.json`,
`review/metadata_provenance.json` and `review/notebook_cpu.json`. Notebook tooling uses the existing
development venv; the experiment retains the separate exact Colab lock from `wam_qualification`.

The CPU fixtures are engineering checks, not DEV/CAL/TEST observations and not new experiment attempts.
The real GPU run remains unexecuted by Codex. No new Gemini review was performed for this package.

## Entry and history

Open `notebooks/latent_control_colab.ipynb`, choose GPU and Run all. Only the user starts it.
It needs the existing sealed Drive path `Model-WM/wam-writer-abc-20261115-attempt01/summary.json`;
no user-uploaded files or manual parameters are needed. Output is `Model-WM/latent-control-20261116-attempt01/`.

The inherited observation/debit floors are preserved in `historical_floor.json`, and the current sealed
Drive receipt can only raise them. Runtime/cache checks use current files and imports. All new and
restored pretrained payloads remain under /content. Prior Drive caches are read-only; no old attempt
is cleaned or rewritten. Pretrained content history stays UNVERIFIED and historical CPU stays a lower bound.

Preparation has its own1800s wall envelope. Model loading to sealing has21600s including60s reserve;
totalCPU43200 core-s, RAM32GiB, VRAM16GiB, results10GiB. All installation, generation, writing, evaluation,
archival and unknown-tail reserves remain in complete cumulative accounting. Closed failure summaries
are read-only on reopening; an unsealed reservation does not allow resume or a fresh model trial.

Real checkpoint compatibility, signal quality, JPEG robustness and resource feasibility remain to be
measured in the single user-started attempt. Numeric success cannot automatically resolve visual
review or establish scientific novelty.
