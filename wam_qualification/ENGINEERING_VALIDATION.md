# 2026-11-14: same-attempt local receipt repair

48 CPU tests and complete pinned synthetic loading passed; actual saved failure-receipt admission and final-volume equality also passed. See [scope and evidence](LOCAL_RECEIPT_RECOVERY.md). Frozen scientific inputs and phase budgets are unchanged. No real weights or GPU; no new Gemini audit.

# Separately budgeted execution — 2026-11-12

Current delivery uses `separated_entry.py`.39 CPU regressions passed, including actual descendant-process cancellation, final CPU reconciliation, phase separation, cumulative preparation recovery and blocked hard-loss recovery. The full fresh-process synthetic WAM/SD asset-preparation and Backend loading check also passed; no real checkpoint or GPU run was performed. Exact frozen manifest/protocol/backend/engine/worker/locks/source-history checks remain unchanged.

**Gemini did not review this revision.** agy returned individual-quota exhaustion before model inference. The user explicitly waived this release gate and requested publication with that status disclosed. See [Gemini status](review/separated_gemini_status.json), [CPU validation](review/separated_cpu_validation.json), [39 test log](review/separated_cpu_tests.log), [synthetic loading](review/separated_full_loading.json), and [additive historical correction](review/separated_history_correction.json). Earlier Gemini approvals below apply only to earlier revisions.

Budget basis is existing real Colab cold-entry evidence, not a new local-throughput estimate: a completed cold entry used353.79052996635437s (including later asset work and the pre-load import failure), while the maximum completed cold-entry CPU observation was199.42112799999998core-s. Three times these values, rounded upward to a minute, yields preparation1080wall/600CPU. Model loading and fixed operations receive the original-sized900GPU/1800wall/600CPU envelope. This is a user-authorized new execution budget, not a reset or PASS of attempt01. Global budget debits remain enforced; all failures are retained.

Historical CPU is irrecoverably imprecise after the old process-group kill. The highest observation adds at least59.742005core-s to the regressed final receipt; cumulative measured CPU is at least2404.914067. For global admission the old WAM600CPU allowance remains fully debited, producing2562.099122 historical budget debit. The157.185055 difference is a reservation, not invented measured consumption. New stage owners seal descendant CPU and include owner exit accounting; abrupt losses without sealed receipts block automatic continuation.

The notebook is a user-run GPU entry. CPU checks do not establish real model/CUDA validity, quality or completion. Full raw engineering logs and the quota rejection are retained under the local `audits/wam-separated-budget-20261112/` directory.

Historical validation records follow.

---

# Technical recovery 02 — 2026-11-12

Current repair fixes the confirmed inherited Colab `MPLBACKEND` failure by setting Agg only in supervised children and checking official imports before loading. The same attempt receives one new, exclusive technical-recovery-02 segment after validating both zero-operation failure receipts. Actual cumulative resources and unchanged historical files were checked locally; no original allowance is reset.

26 CPU regressions, pinned dependency checks, and the full fresh-process synthetic loading preflight passed. The latter exercises real asset preparation, local cache restoration, strict official WAM deserialization, real tiny SD save/from_pretrained, actual Backend initialization, DPM scheduler conversion, FP32 and official image transforms. CUDA availability and transfer are redirected for this CPU test; no real checkpoint, real generation or GPU execution was performed. Tiny SD fixtures do not certify real-model memory, quality or CUDA compatibility.

Gemini through agy (`gemini-3.8-flash-high`) read all four bounded source bundles, covering 20 source/config/test files including the notebook builder, and returned `PASS_FOR_USER_RUN_WITH_FIXED_BUDGET`. See [review](review/gemini_recovery02_20261112.txt), [CPU summary](review/recovery02_cpu_validation.json), [full loading receipt](review/recovery02_full_loading.json), and [unit test log](review/recovery02_cpu_tests.log). The [final supplement](review/gemini_recovery02_cpu_final.txt) also returned PASS for the strengthened actual asset-preparation CPU path; [provenance](review/gemini_recovery02_provenance.json) records exact reviewed code and limitations. The earlier rejected model/effort configuration and every local diagnostic failure remain archived.

New payload downloads and restored weights live in `/content`; existing paid Drive caches are read only. Copy time, setup, downloads, CPU work, model loading and execution consume the remaining original allowance. CPU engineering checks are separately recorded, not counted as experiment trajectories or scientific success. The notebook may validly stop on budget, source or quality checks. No training or CAL/TEST.

Historical release records follow.

---

# Technical-recovery validation — 2026-11-12

Gemini via agy completed the recovery patch review with no confirmed blocking/accounting defect; see [original review](https://github.com/RICHAAARC/Model-WM/blob/main/wam_qualification/review/gemini_recovery_20261112.txt) and [provenance and limits](https://github.com/RICHAAARC/Model-WM/blob/main/wam_qualification/review/gemini_recovery_provenance.json). Earlier timeouts remain retained.

Original execution stopped before any of240 operations because the Meta leaf certificate expired. Its final cumulative receipt is retained and carried forward. The recovery patch passed20 CPU regressions (11 original+9 added), dependency checks,8 official synthetic-state/interface checks, and14 verified HEAD probes without weight bodies. Actual failure-receipt admission and frozen input/lock byte comparisons also passed. Full logs are retained in the workspace `audits/wam-tls-recovery-20261112/`. No real checkpoint or GPU execution occurred during repair. The following original-release record is historical.

---

# Release validation — 2026-11-10

Gemini via agy1.5.4, model `gemini-3.8-flash-high`, completed the full code review. Final conversation `8770379f-1f5a-4891-b24b-8ccb88afc467` returned SUCCESS with a nonempty final review and no denied action in the final turn. All runtime Python, protocol, tests, dependency locks, notebook builder, preregistration and source-history text were supplied in three bounded source bundles. Frozen full bit arrays were checked locally without redraw. The final review found no confirmed blocking implementation defect; it did not certify real model/GPU behavior.

The initial builder-path finding was rejected: the reviewer copy had flattened the script's original `tools/` directory. In the real repository `parents[1]` is correct, and the actual builder exits0. No speculative change was made to working runtime logic. The review workflow was corrected to retain original source paths, check actual read coverage, require a nonempty final response, inspect denied actions, and persist native CLI output immediately. High-reasoning work continued after earlier CLI timeouts; incomplete-step usage was absent from those partial receipts. Earlier timeout, empty-success and denied-read evidence remains in the local audit archive, not overwritten.

CPU checks on the exact pinned Python3.10.14/torch2.5.1+cpu environment:

- `pip check`: passed.
- Official architecture with synthetic state: strict loading, missing/mismatched state rejection, exact512-pixel embed equality,33-channel detect,4x4 mean preservation, official .5 native threshold and empty-mask handling passed. No real checkpoint was read.
- All11 bookkeeping/media/budget/failure regressions passed. The full mock path uses the fixed16/64/160 counts and does not become scientific evidence.
- The existing official Python3.10.14 archive unpacked and executed `--version` successfully on CPU, without a new download.
- Frozen `manifest.json`, `protocol.py`, `source_usage_history.json` remain byte-identical to the pre-audit snapshot. Other runtime Python and the dependency locks remain unchanged from the prepared package.

The latest actual Drive cumulative receipt was fetched again: conservative GPU/wall2500.71440577507s, CPU1962.099122 core-s, downloads10791365453bytes, old results41151509bytes, old trajectories44/44. Runtime still rereads that latest receipt and carries any higher non-regressing costs; it does not reset a budget or a trajectory.

Release CPU command group used45.7788s wall/41.638371 CPU core-s as local engineering work, separately recorded from the not-yet-started qualification. Actual qualification operations remain0. Local audit folder: `audits/wam-release-audit-20261110/` in the research workspace.

The notebook is verified structurally and by unpacking its exact embedded code. Its real Run-all model stage has deliberately not been executed. The user's next real execution requires one Colab GPU and the existing Drive history. Cold setup may consume the frozen900s conservative allowance; budget/source/quality failures remain stop outcomes. No training, CAL/TEST, FWER or innovation PASS is implied.
