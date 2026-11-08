# Local validation

## Same-attempt model-stage continuation — 2026-11-08

43 CPU engineering checks passed: 39 lightweight controls (14 original, 13 environment-recovery, 12 model-stage) and the 4 unchanged exact-version LoRA regressions. New coverage includes finalized latest-receipt admission, rejection of the stale original cost baseline, frozen-input and version mismatch, missing GPU without model reservation, original-file preservation, failed and hard-interrupted slot counts, duplicate invocation protection, stopping before subprocess launch when cumulative budget is spent, reuse without reinstallation, cache provenance and accounting, software-plus-model cost preflight, unknown asset size, per-download cumulative limits, and metadata-only environment verification.

Additional bounded CPU checks used actual saved Drive receipts to admit the original failure→ENVIRONMENT_READY chain without invoking the device/model path. Real pip with `--no-index` reused a small cached wheel and retained its link, with zero new download bytes. The already acquired official Python archive rebuilt a 3.10.14 interpreter offline, also with zero new download bytes. The frozen manifest, protocol, trajectory ledger implementation, model runner, metrics, production requirements and locked LoRA tests remain byte-for-byte unchanged from `feb3f79`.

One HEAD-only preflight of all 18 frozen official assets returned 4,836,415,535 bytes. No model payload was downloaded. With a live ready environment, this fits the remaining effective 6,735,101,044-byte download allowance; with all software cache lost, the historical software estimate raises the total to 8,012,070,089 bytes and the entry stops before installation/payload downloads. Real runtime monitoring and per-file streaming limits remain authoritative; this preflight does not certify GPU time, compatibility, or completion.

```bash
python -m unittest tests.test_controls tests.test_environment_recovery tests.test_model_stage -v
# Existing exact Python3.10.14 / torch2.0.1+cpu regression environment:
python -m unittest tests.test_lora_loading_locked -v
```

The notebook remains unexecuted, with its first cell containing only Drive mount. It now calls `run_model_stage` and requests a GPU runtime for the user to start. This release task did not change Drive records, run Colab, download/load real model weights, initialize CUDA, generate a trajectory or enter CAL/TEST. CPU delivery is not a scientific or qualification PASS.

## Python acquisition and bounded environment recovery — 2026-11-08 (prior release record retained)

31 CPU engineering checks passed: the updated 14 original controls, 13 recovery/download checks, and all 4 exact-version LoRA loading regressions. The original production manifest, protocol, requirements, trajectory ledger, model runner, metrics and media code remain byte-for-byte unchanged from `9bfdd61`.

The 13 new checks cover exact cumulative resource accounting without a device query; peak-versus-sum semantics; stopping before subprocess launch when the remaining budget is exhausted; byte-preserving original failure/ledger records; one recovery reservation; retained failure/interruption; repeated clicks; rejection of missing/regressed/nonfinite resource evidence, changed manifests and consumed trajectory slots; an environment-only command path; explicit local Python and zero retries; single official API request; retained partial/failed downloads; unsafe tar paths/links; and Python version mismatch.

The actual official Astral Release asset API delivered the same CPython 3.10.14 / 20240814 archive (20,510,166 bytes observed, not an admission-size rule). The extracted interpreter reported Python 3.10.14. Using uv0.9.5 with `--python <explicit path> --no-python-downloads --seed` created a fresh local CPU environment. This verifies the acquisition and interpreter/venv path on the local Linux host, not Colab network availability. Download receipt, actual setup log, CPU results and original failed records are retained in the local audit directory.

```bash
python -m unittest tests.test_controls tests.test_environment_recovery -v
# Existing locked Python3.10.14 / torch2.0.1+cpu environment:
python -m unittest tests.test_lora_loading_locked -v
```

The notebook is unexecuted, Drive-only in its first cell, CPU by default, and calls only `recover_environment`. Successful recovery stops at ENVIRONMENT_READY, with original attempt01 costs carried forward. No Drive attempt was changed during local validation. No real model weights, model generation, CUDA initialization or Colab runtime were used. The production CUDA dependency installation is still a user-run Colab environment check; the locked CPU regression does not certify it. No scientific PASS is claimed.

## LoRA interface correction — 2026-11-08 (prior release record retained)

18 engineering checks passed: 14 lightweight controls (the original 13 plus a real local-Git checkout/cache update check) and 4 locked CPU LoRA regressions. No real checkpoints, image generation, Colab execution or CUDA initialization were used.

The conversion context is now `torch.no_grad()` so Diffusers can wrap its ordinary tensors as Parameters. The conversion algebra, values and scales are unchanged; generation and extraction keep their existing `inference_mode()` contexts. The locked regressions cover all four conversion branches (attention, feed-forward, proj_in, proj_out), reproduce the old failure as an expected negative control, load/fuse/unload real Diffusers layers, and restore the clean base across multiple synthetic messages without accumulation.

The regressions require Python 3.10.14, torch 2.0.1+cpu, torchvision 0.15.2+cpu, diffusers 0.24.0, transformers 4.26.1, accelerate 0.28.0, huggingface-hub 0.21.4, safetensors 0.4.2 and NumPy 1.24.3. A version mismatch fails the check instead of skipping it. These CPU builds test the loader semantics; they do not certify the cu118 build or a real GPU.

Run from the repository root, using the corresponding local environments:

```bash
# Lightweight environment: NumPy, Pillow, scikit-image, psutil, nbformat, Git.
python -m unittest discover -s tests -p test_controls.py -v
# Locked Python 3.10.14 CPU environment; no downloads during the test.
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m unittest discover -s tests -p test_lora_loading_locked.py -v
```

To provision a separate CPU check environment, use `requirements-colab.txt` with only the two `+cu118` wheel suffixes replaced by `+cpu`, and the PyTorch CPU wheel index. Do not change the production requirements. Add nbformat 5.10.4 only if also running the notebook controls in that environment. The CPU environment's `pip check` passed.

The notebook now fetches and fast-forwards an existing checkout and reloads its bootstrap module, so a prior Colab session cannot silently retain the old source. It does not reset local edits or alter existing Drive results. The first cell remains Drive-only and all cells remain unexecuted. The frozen manifest, protocol, production dependencies, trajectory ledger, budget supervisor, media operations and acceptance gates are unchanged from initial commit `280dbd348b991ea4edbe9afc0ab8585515f824e0`.

Before-fix failures, the previous preflight record, and after-fix results are retained separately. A completed CPU check permits release of this interface fix; real weight loading, GPU memory/throughput and Drive runtime behavior remain unverified. No scientific PASS is claimed.

## Initial publication record (retained)

Engineering checks only; no real model weights downloaded or loaded, no GPU generation, no Colab execution.

13 unittest checks passed on Python 3.12 with NumPy 1.26.4, Pillow 10.2.0, scikit-image 0.22.0 and nbformat 5.10.4. Colab uses the separately pinned upstream inference stack on Python 3.10; its execution remains unverified.

Checked: 44-slot fixed schedule, shared work identity/noise, continuous code slices, fixed probe flips, durable reservation, duplicate-run protection, fixed denominators, integer pixel blending, PSNR/SSIM scale, unresolved zero probe denominator, budget margin, actual child termination on injected budget stop, failed-child no retry, rejection of nonfinite evidence, first Drive-only cell and notebook format/syntax.

No dependency/GPU compatibility or scientific performance PASS is claimed. Real run failures stop and remain in Drive.
