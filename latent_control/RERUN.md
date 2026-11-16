# User-authorized larger-GPU rerun — 2026-11-16

The user explicitly requests a full rerun on a different GPU and deletion of the old failure output.
`notebooks/latent_control_rerun_colab.ipynb` is the new upload-free entry. The GPU remains user-started.

- Attempt02 has a new execution allowance at the same preparation/model/CPU/storage limits.
  GPU memory limit follows the detected single device, which must exceed 16 GiB. A100 is preferred;
  it has not been validated with real weights by this delivery. Method memory management is unchanged.
- Works, codebook, keys, seed, float32/512px, samplers, algorithms, operation limits and all
  scientific gates are unchanged. CAL is sealed before TEST. This is a rerun, not fresh independent DEV:
  DEV/000 was previously generated; CAL/TEST were not entered.
- Before deleting exactly `latent-control-20261116-attempt01`, the entry reads its sealed receipt
  and saves/readbacks aggregate costs and operation counts in `latent-control-20261116-attempt02-carry.json`.
  Old logs, traceback, outputs and source snapshot are deleted from that Drive folder. Other experiment
  directories are untouched. Produced-result volume remains cumulative, not current Drive occupancy.
- New output is `latent-control-20261116-attempt02`. Reopening a reserved or completed attempt is
  read-only. An interruption during cleanup can reuse the aggregate carry receipt; it cannot refund
  old cost or resample data. Local paid caches can be reused if they still exist; they are not deleted.
- The notebook itself performs cleanup when the user runs it. Publishing does not delete Drive files,
  load pretrained weights or execute GPU work. New run results are recorded normally.

CPU checks cover bounded deletion, symlink rejection, carry-before-delete and interruption, immutable
method fields, GPU preflight, repeat-click lockout, and actual notebook extraction/import/read-only path.
These are engineering tests, not evidence of real GPU memory feasibility or scientific success.
