# SD2.1 × LaWa matched-loss comparison
The completed 8-work DEV passed the predeclared structural-loss matching gate and AI engineering image screen. Ordinary gain is 1.025; fixed coarse transfer remains 0.10. The DEV ordinary/coarse loss ratio is 0.9870254858387113; this is tolerance-matched structure loss, not exact perceptual equality.

Open notebooks/lawa_sd21_matched_compare_colab.ipynb in Colab, select a GPU and Run all. No source upload is required. Keep the existing lawa-sd21-matched-dev-20261125-attempt01 folder in the same MyDrive/Model-WM directory. The entry reads its final receipt, quality records and frozen allocation before preparing any model work.

The 24 CAL / 80 TEST rows, generation seeds and identity pairs are exactly the corresponding rows from the original sd2_matched/allocation.json. No new draw occurs. Full original preregistration is retained verbatim in PREREGISTRATION.md; its “DEV-only notebook” wording describes the earlier entry. This package implements its already-registered later stages. All thresholds and inference rules are reused through matched_compare.worker and matched_compare.rules; only the generation backend and independent data/calibration change.

Fixed work: 104 generation / 1248 write / 1248 mix / 5200 read. CAL completes and its shared threshold is sealed/read back before TEST. Equal-three PNG is the sole primary endpoint. No new samples, training, seed changes or automatic reruns. Completion, negative outcomes and failures are retained. The SD1.4 result remains separate.

Construction: SD2.1 community-mirror denoiser plus the fixed SD1.4 clean KL-f8/LaWa decoder. This is cross-denoiser replication, not an untouched native SD2 pipeline or a second watermark method.

Preparation, model and settlement are metered independently; both historical ledgers inherit the completed DEV. There is no preparation/model total-time cap. Estimates are preparation 5–20 minutes and model 70–120 minutes, not deadlines. Current validated /content caches may be reused; replacement runtimes require preparation. Weights and environment remain outside Drive.

Output: MyDrive/Model-WM/lawa-sd21-matched-compare-20261125-attempt01.
Repeated execution reads existing receipts and cannot repeat started model work.
CPU tests use synthetic fixtures, not pretrained tensors. GPU execution is initiated only by the user.
