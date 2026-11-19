"""Resume sealed works without replaying an interrupted physical operation."""
import sys
import traceback
from pathlib import Path

import torch

from latent_control.io import Ledger, save
from latent_control.protocol import arrays
from terminal_solver import quality, science
from terminal_solver.model import Solver
from terminal_solver.protocol import limits, protocol, read_json
from terminal_solver.worker import generate_work
from wam_method.entry import publish_closed
from .checkpoints import bundle, validate_counts
from .entry import latest_session


def run(local, output, manifest, load_model, generate=generate_work):
    selected = read_json(output/'migration.json')['gains']
    gains = {arm: [gain] for arm, gain in selected.items()}
    model = None
    all_counts = {}
    for split in ('CAL', 'TEST'):
        works = [w for w in manifest['works'] if w['split'] == split]
        data = {arm: [] for arm in 'CO'}
        missing = []
        for work in works:
            if local.name.isdecimal() and latest_session(output/'invocations').name != local.name:
                raise RuntimeError('Superseded invocation; no new work may start')
            wid = work['id']
            key = wid.replace('/', '-')
            receipt = output/'work_receipts'/(key+'.json')
            claim = output/'reservations'/key
            if not receipt.exists():
                if split == 'CAL' and (output/'calibration.json').exists():
                    missing.append(wid)
                    continue
                if claim.exists():
                    missing.append(wid)
                    continue
                if model is None:
                    model = load_model()
                claim.parent.mkdir(parents=True, exist_ok=True)
                try:
                    claim.mkdir()
                except FileExistsError:
                    missing.append(wid)
                    continue
                save(claim/'start.json', dict(id=wid, invocation=local.name))
                save(local/'progress.json', dict(phase=split, active_work=wid,
                     evaluation_bundles=sum(len(v) for v in data.values())))
                work_log = local/'journals'/key
                ledger = Ledger(work_log, limits([work]))
                model.ledger = ledger
                try:
                    timing = generate(model, work, gains, local, output, ledger, manifest)
                    save(claim/'timing.json', timing)
                except BaseException:
                    (claim/'exception.txt').write_text(traceback.format_exc())
                    media = local/'works'/wid
                    if media.exists():
                        publish_closed(media, claim/'partial')
                    raise
                finally:
                    publish_closed(work_log, claim/'journal')
            # The original writer issues this receipt only after both arms are on Drive.
            record = read_json(receipt)
            if record['id'] != wid or not claim.is_dir():
                raise ValueError('Invalid work reservation or seal: '+wid)
            validate_counts(record['operations'], limits([work]))
            for arm in 'CO':
                data[arm].append(bundle(output, work, arm, selected[arm], verify_images=True))
            for kind, n in record['operations'].items():
                all_counts[kind] = all_counts.get(kind, 0) + n
        if missing:
            return dict(status='INCOMPLETE_'+split, missing=missing, completed=len(works)-len(missing),
                        denominator=len(works), operations_completed_in_sealed_works=all_counts)
        if split == 'CAL':
            computed = {arm: science.calibrate(data[arm], expected=64) for arm in 'CO'}
            seal = output/'calibration.json'
            if not seal.exists():
                save(seal, computed)
            cal = read_json(seal)
            if cal != computed or not all(c['sealed'] for c in cal.values()):
                raise ValueError('CAL seal does not match the complete frozen CAL records')
    result = science.verdict(data, cal)
    validate_counts(all_counts, limits([w for w in manifest['works'] if w['split'] != 'DEV']))
    save(output/'paired_results.json', result)
    quality.gallery(output, works, selected, result)
    return dict(status='PROXY_SCREEN_PASSED' if result['passed'] else 'GATE_FAILED',
                completed=dict(CAL=64, TEST=64), operations=all_counts)


def main():
    local, output = map(Path, sys.argv[1:3])
    manifest = protocol()
    torch.set_num_threads(1)
    torch.manual_seed(manifest['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False

    def load_model():
        return Solver(read_json(local/'model_paths.json')['base'], manifest, arrays(), None)

    save(local/'outcome.json', run(local, output, manifest, load_model))


if __name__ == '__main__':
    main()
