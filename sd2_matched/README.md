# SD2.1 / LaWa DEV delivery

[Open Run-all Colab](https://colab.research.google.com/github/RICHAAARC/Model-WM/blob/main/notebooks/lawa_sd21_matched_dev_colab.ipynb)

Select a GPU, then Run all. The first cell only mounts Drive; the notebook contains its full code bundle. No uploads or editable experiment parameters are required. It performs 8 generations, 240 writes and 24 interface reads on the first two scheduled DEV works. It cannot run CAL/TEST. Both the 112-work allocation and 8/24/80 split were drawn once and frozen before any generation.

Output: `MyDrive/Model-WM/lawa-sd21-matched-dev-20261125-attempt01/`.

Expected completion: `DEV_MATCHING_GATE_MET`, `interface_numeric_passed=true`, `visual_review=PENDING`, `scientific_pass=null`. `DEV_MATCHING_STOP` means the fixed gain grid failed quality matching; `DEV_STOPPED_INCOMPLETE` preserves a technical failure. Reopening reads the saved attempt and does not retry. Return `summary.json` or its printed receipt after completion; the front two works and all DEV images must be reviewed before the next notebook is released.

Weights stay only in `/content`. Current-runtime environment and source-validated local caches are reused; runtime replacement may require downloads again. Preparation/model/settlement costs inherit both historical ledgers from the completed SD1.4 comparison. There is no arbitrary global wall-time cutoff. Estimated DEV preparation 5–20 minutes and model 5–15 minutes are estimates, not stop limits. All started operations, failures and costs remain recorded.

This is a community-mirrored SD2.1 text encoder/denoiser combined with the unchanged clean SD1.4 VAE and LaWa writer/reader. See SOURCE_AUDIT.md. It tests cross-denoiser replication, not a new watermark method or cross-watermark generalization.

CPU evidence: 12 unit/integration checks, actual component-config parsing and meta architecture, synthetic component loading plus 50-step DDIM at 512 pixels, official 484-state LaWa strict synthetic load and reduced-resolution writer/reader execution, unchanged uniform-hook regression, notebook-schema and embedded-source checks. No pretrained weights or GPU were used. See ENGINEERING_CHECKS.json. Actual real-weight/GPU/image compatibility remains for the user-run DEV.

Commands from repository root (pinned CPU environment):
```
python -m unittest sd2_matched.test_cpu -v
python -m sd2_matched.check_official_cpu /path/to/pinned/LaWa/source
python -m unittest matched_loss.test_cpu.Checks.test_uniform_tensor_hook -v
```

Frozen SD1.4 scientific evidence and manuscripts are unchanged. Source_evidence retains public configuration and documentation snapshots; these are not model weights.
