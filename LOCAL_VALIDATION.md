# Local validation

Engineering checks only; no real model weights downloaded or loaded, no GPU generation, no Colab execution.

13 unittest checks passed on Python 3.12 with NumPy 1.26.4, Pillow 10.2.0, scikit-image 0.22.0 and nbformat 5.10.4. Colab uses the separately pinned upstream inference stack on Python 3.10; its execution remains unverified.

Checked: 44-slot fixed schedule, shared work identity/noise, continuous code slices, fixed probe flips, durable reservation, duplicate-run protection, fixed denominators, integer pixel blending, PSNR/SSIM scale, unresolved zero probe denominator, budget margin, actual child termination on injected budget stop, failed-child no retry, rejection of nonfinite evidence, first Drive-only cell and notebook format/syntax.

No dependency/GPU compatibility or scientific performance PASS is claimed. Real run failures stop and remain in Drive.
