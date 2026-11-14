# attempt02 engineering closure 01

Authorized 2026-11-14: finish the three engineering repairs, add preparation
budget from actual cold startup observations, publish a user-started entry.
No new method, seed, codebook, work, extra trial, GPU run by the agent, or CAL/TEST.

The same attempt02 retains invocations 0001–0004 byte-for-byte. Invocation 0005
is reserved once by `engineering_closure_01.started.json`, even if preparation
fails. A second click reads the sealed result. An active/unsealed reservation
blocks execution. All 240 operation rows must be NOT_STARTED before admission.
The original model reservation from 0003 remains; a separate closure model
reservation is created before loading weights. No model operation can retry.

## Additive preparation budget

The original `execution_budget.json` is unchanged, both in source and Drive.
The amendment and its effective budget are saved with invocation 0005.

| Envelope | Original | Added | Effective cumulative cap |
| --- | ---: | ---: | ---: |
| Preparation wall seconds | 1080 | 1560 | 2640 |
| Preparation CPU core-seconds | 600 | 660 | 1260 |
| Model GPU allocation proxy / wall / CPU | 900 / 1800 / 600 | 0 | 900 / 1800 / 600 |

Basis: invocation03 spent 503.1773302555 wall seconds, including 8.3532180786
model seconds: preparation was 494.8241121769 seconds. Invocation04 installed
its cold environment in 342.815919853 seconds; the earlier successful asset
stage took 142.864191944 seconds. The maximum observed completed cold entry
CPU cost is 199.421128 core-seconds. Added wall = round up to a minute
`3 * 494.8241121769 + 60` = 1560 seconds. Added CPU = round up
`3 * 199.421128 + 60` = 660 core-seconds. These are estimates with reserve,
not completion guarantees. All global cumulative caps remain unchanged.

At the audited 0004 receipt, preparation debits are 1082.1840512753 wall and
412.111483 CPU, leaving 1557.8159487247 wall and 847.888517 CPU under the amended
cap, before new entry overhead. Model debits remain 28.3532180786 wall/GPU
proxy and 28.044591 CPU. Historical reserves and unknown CPU/sealing tails
remain explicitly distinguished from measured lower bounds; nothing is refunded.
The old 1080-second envelope did not pass and is not retroactively relabeled.

## Startup and recovery

Before installation or checkpoint copies, the host fetches the small base
metadata from the fixed official revision and verifies JSON/component closure.
After the pinned environment validates, the actual tokenizer, processor,
scheduler and model-config parsers run before checkpoint bodies are restored.
Known safety-checker disabling is unchanged; the processor is never set to None
to mask missing dependencies. Source configuration/import checks precede WAM
checkpoint restoration. All model tensors remain under /content.

Before expensive work, the remaining preparation AND global budget must cover
1080 wall / 480 CPU for a conservative cold start. Before asset restoration it
must cover 420 wall / 120 CPU. The gates include cleanup reserve and use
conservative debit fields, never just measured lower bounds. An existing Python
path is not treated as a validated environment or a reason to waive this gate.

The existing environment is probed for pinned versions in the current runtime;
compatible installations are reused, partial installations are repaired using
paid packages. Download cache, installed environment, and in-memory models are
different: a new Colab runtime loses /content, installations and memory; only
existing durable artifacts can be restored. Existing Drive caches are read-only;
new weights/packages are not saved to Drive. Cache hits still incur copy and
installation time. No check creates a new cache archive on Drive.

Live communication remains local; only quiescent receipts/results cross Drive.
The stopping margin is 60 seconds: 40 for cancellation/ordinary settlement and
20 prepaid for the terminal receipt. Bulk and ordinary IO are measured. Terminal
IO is bounded and verified before return; overrun retains the lock and marks
accounting unresolved. This is a budget reserve, not a fabricated measurement
or a guarantee against arbitrary Drive stalls. Terminal success duration is
not persisted as an exact historical measurement.

## Evidence ceiling and first real work

Release checks: 63 CPU regressions passed, including concurrent receipts,
descendant cancellation, delayed sealing, four-failure admission and read-only
reopening. The actual saved receipt fixture preserved 70 selected old files.
The three-cell notebook's 36 embedded source files match this release. Frozen
manifest, protocol, Backend, engine, worker, core, dependency pins, original
budget and preregistration are unchanged from ae2ea2a.

CPU validation uses actual pinned loaders, the official component index and
processor config, exact official WAM architecture with synthetic tensors and a
tiny random SD fixture. The real worker and Backend generate/write/extract run
through the first whole work, including PNG/JPEG, mixes, continuous logits,
blind scores, projection and ledger. Only devices and the boundary before work
two are redirected; model functions are not replaced. The production manifest
is unchanged. CPU fixture output is engineering evidence, not an experiment.

The legal first real unit remains 1 generate -> 4 writes -> 10 extracts, already
within 16/64/160. The existing worker then predicts the remaining cost; if it
fits, the same runtime continues the same ordered qualification. There is no
extra warmup or alternate order. Real checkpoint/CUDA compatibility, runtime
cost and numerical/visual qualification remain unverified until the user runs.
No new Gemini audit was performed for this engineering closure.
