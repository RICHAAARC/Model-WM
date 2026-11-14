# Current entry: attempt02 technical continuation 01

See [scope, accounting and CPU evidence](TECHNICAL_CONTINUATION_01.md). One user-started continuation of 0001–0003; next invocation 0004; all phase budgets and physical operation limits remain unchanged. Older records below are retained history.

---

# Current repair: attempt02 local receipts

See [local receipt recovery](LOCAL_RECEIPT_RECOVERY.md). Reopen the published notebook to continue the same attempt02, preserving 0001/0002 and every budget debit. Live communication is under `/content`; Drive receives sealed records and results. GPU execution remains user initiated.

---

# Current entry: separately budgeted attempt02

See [separate preparation and model execution](SEPARATED_EXECUTION.md). The notebook targets `wam-qualification-20261112-attempt02`; attempt01 is retained as an environment-stage stop. The user starts GPU execution. Prior records below describe superseded entries.

---

# Current entry: technical recovery 02

See [technical recovery 02](TECHNICAL_RECOVERY_02.md). This is the only current continuation target. Weights and new downloads live under `/content`; Drive stores outputs and receipts. Both previous failures and their costs remain retained. The older release records below are historical.

---

# Same-attempt technical recovery — 2026-11-12

The revised notebook targets `wam-qualification-20261109-attempt01/technical-recovery-01/`. Keep the original Drive output and caches. Select one GPU and Run all once; no code upload is needed. A live compatible WAM environment is reused; otherwise retained pinned wheels install offline. The first cell mounts Drive; resource checks precede installation. All new work consumes the original remaining budget. Reopening the recovery is read-only. See `TRANSPORT_RECOVERY.md` for exact admission, provenance and remaining caps.

The prior full Gemini review below describes the original package. The recovery patch has a separate review record. CPU checks and HEAD availability do not establish real checkpoint/CUDA success or full completion within the remaining time.

---

# WAM qualification entry

Delivery status (2026-11-10): Gemini via agy completed the full code review and found no confirmed blocking implementation defect. The earlier builder-path finding was caused by the review copy's layout; the actual script is in `tools/` and builds correctly. See the [review](https://github.com/RICHAAARC/Model-WM/blob/main/wam_qualification/review/gemini_final.txt) and [provenance](https://github.com/RICHAAARC/Model-WM/blob/main/wam_qualification/review/provenance.json). All earlier timeout and denied-read receipts remain archived. Real weights and GPU execution remain untested; this review is not an experimental PASS.

Open the companion `notebooks/wam_qualification_colab.ipynb`, select one GPU and Run all. The notebook contains the frozen lightweight package; no code/file upload is requested during its execution. The first cell only mounts Drive. The second starts the meter, unpacks the embedded package and enters the once-only qualification. Actual model execution is user initiated.

Prerequisites: the same Google Drive with `Model-WM/aqualora-qualification-20261108-attempt01/rerun-02/latest.json`, standard Colab `psutil`, and one CUDA12.4-compatible GPU. Existing paid SD1.5/Python caches are reused when their source receipts are readable. A cold setup may consume the15-minute allowance before model execution; the notebook stops rather than silently extending it.

Output: `MyDrive/Model-WM/wam-qualification-20261109-attempt01/`. Inspect `summary.json`, `operation_status.json`, `readouts.jsonl`, `native_quality.jsonl`, `paired_diagnostics.json`, `qualification_projection.json`, all sixteen `visual_review` panels, and failure/setup logs. Reopening never reruns or resumes.

This is a carrier/interface/quality/cost qualification, not the catch-one learning experiment. No training, CAL/TEST, FWER claim or innovation PASS. See `PREREGISTRATION.md` for exact definitions and retained history. Engineering validation and Gemini audit are not experimental results.
