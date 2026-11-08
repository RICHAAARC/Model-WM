# Local validation

## LoRA interface correction — 2026-11-08

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
