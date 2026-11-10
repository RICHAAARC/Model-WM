"""Single model worker with measured conservative work projection."""
import json
import sys
import time
from pathlib import Path
from .core import save
from .backend import Backend
from .engine import run


def main(root):
    root = Path(root)
    # Exclusive worker reservation also prevents accidental double invocation.
    with (root/"worker_started.json").open("x") as f:
        json.dump(dict(started_unix=time.time()), f)
    manifest = json.loads((root/"manifest.json").read_text())
    paths = json.loads((root/"model_paths.json").read_text())
    entry = json.loads((root/"runtime_entry.json").read_text())
    backend = Backend(manifest, **paths)

    def budget(index, timings):
        if not timings:
            return  # First work measured once, never an extra warm-up trajectory.
        s = json.loads((root/"resources.json").read_text())["resources"]
        limits = entry["effective_limits"]
        remaining = 16-index
        wall = max(x["wall_seconds"] for x in timings)*remaining
        cores = max(x["cpu_seconds"] for x in timings)*remaining
        ok = (wall <= .75*(limits["gpu_seconds"]-s["gpu_occupancy_seconds"]-20)
              and wall <= .75*(limits["wall_seconds"]-s["wall_seconds"]-20)
              and cores <= .75*(limits["cpu_seconds"]-s["cpu_seconds"]-20))
        save(root/"qualification_projection.json", dict(works_observed=index, works_remaining=remaining,
            projected_remaining_wall_seconds=wall, projected_remaining_cpu_seconds=cores,
            remaining_budget_reserve=.25, admitted=ok, note="maximum observed whole-work cost; no uncounted warm-up"))
        if not ok:
            raise RuntimeError("Remaining fixed qualification projects outside budget plus 25% reserve")
    run(root, manifest, backend, budget, precreated=True)


if __name__ == "__main__":
    main(sys.argv[1])
