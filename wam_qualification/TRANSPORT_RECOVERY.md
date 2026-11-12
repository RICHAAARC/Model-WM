# WAM same-attempt technical recovery — 2026-11-12

The user explicitly approved one technical recovery activation and publication after Gemini review, with real GPU execution remaining user initiated. This amendment changes transport and recovery control only. All frozen codebooks, seed, parameters, source history, quality gates,16/64/160 physical limits and original budgets remain unchanged. It does not authorize extra model retries, training or CAL/TEST.

## Confirmed failure and official transport

The Meta server leaf certificate for `*.fbaipublicfiles.com` expired at2026-11-11 23:59:59 UTC (2026-11-12 07:59:59 Asia/Shanghai). The original Colab request received zero checkpoint bytes; environment installation and official source download had completed. A separate local OpenSSL check confirmed leaf-certificate expiry.

The publisher's [pinned model card](https://huggingface.co/facebook/watermark-anything/blob/88ffb2cd1079a76b01d6beda8bf2a433c25fa881/README.md) identifies the MIT/SA-1B release and supplies `facebook/watermark-anything/checkpoint.pth` as its Hugging Face download. The [official code README](https://github.com/facebookresearch/watermark-anything) supplies the same repository/file. The model card's later example names `wam_mit.pth`, but the actual pinned listing contains only `checkpoint.pth`; this entry uses the published file, not the nonexistent example filename. Revision88ffb2cd1079a76b01d6beda8bf2a433c25fa881 was last modified2025-07-23. Its reported size377825938 matches the old Meta HEAD. These support publisher provenance, not a measured bytewise equality between checkpoints. No real checkpoint was downloaded or loaded during repair. Actual strict state loading and unchanged quality gates remain mandatory.

The manifest retains the original logical checkpoint URL. Only that exact URL maps to the pinned official Hugging Face transport; download receipts retain both URLs. TLS verification is enabled, with no insecure transport, alternate model or retry loop. All14 official resource HEAD requests returned200 during local verification without reading weight bodies. Runtime checks source receipts for paid cache and HEAD for missing assets before environment installation. A successful HEAD cannot guarantee later full transfer or CUDA compatibility.

## Accounting and admission

Original `wam-qualification-20261109-attempt01/` files remain unchanged. The entry writes a single `technical-recovery-01/` child. Admission requires the sealed specific zero-byte certificate failure, exact frozen manifest, all240 ledger rows still NOT_STARTED, no prior worker reservation, matching summary denominators, unchanged locks and a successful environment receipt. Missing/inconsistent/nonfinite/regressed costs or consumed operations stop. A later changed Aqua receipt requires reconciliation.

Carried cumulative GPU/wall is2755.732841730118s, CPU2161.52025 core-s, downloads14014744427bytes and results41916035bytes. WAM already spent255.0184359550476 GPU/wall seconds and199.421128 CPU core-s. Remaining WAM caps are644.9815640449524 GPU-clock seconds,1544.9815640449524 wall seconds and400.578872 CPU core-s, additionally bounded by the original global remainder. Existing20s stop margin and25% work-projection reserve remain. Setup, network checks, installation, loading and IO consume the remainder. Paid cache volume is recorded before work and only growth is newly charged. Partial files and failures remain retained.

A live compatible WAM environment is reused. A replaced runtime uses retained pinned wheels offline; missing packages fail rather than trigger an install retry. An absent paid Python archive may be fetched once from its original official source and charged. No new seed, candidate, warm-up or replacement work is added. Reopening the technical segment is read-only, including after interruption. There is no recovery-02 or additional model attempt.

## Evidence ceiling

The preceding CPU checks cover transport failure/no retry, cache reuse, historical reconciliation, consumed-operation refusal, exhausted budgets, entry ordering, preserved originals, offline-wheel recovery and repeat-click behavior. Existing official synthetic state and fixed-media checks remain applicable. Real-state compatibility, download throughput, GPU memory, quality and completion within the remaining budget are not established by these checks. Only the user's bounded run can supply that evidence. A stop outcome remains valid; this notebook does not promise experimental PASS.
