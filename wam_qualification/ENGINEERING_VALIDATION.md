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
