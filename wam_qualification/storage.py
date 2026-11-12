"""Read old paid Drive caches; all new payloads and restored weights live locally."""
import json
import shutil
from pathlib import Path
from .core import append


def locations(root, target):
    target = Path(target)
    receipt = Path(root)/"local_storage.json"
    if receipt.is_file():
        for row in json.loads(receipt.read_text())["mappings"]:
            try:
                relative = target.relative_to(row["new"])
            except ValueError:
                continue
            return target, Path(row["drive"])/relative, Path(row["paid"])/relative
    return target, None, None


def locate(root, target):
    new, drive, paid = locations(root, target)
    local=[]
    metadata=Path(root)/"local_storage.json"
    if metadata.is_file():
        for row in json.loads(metadata.read_text())["mappings"]:
            try:relative=Path(target).relative_to(row["new"])
            except ValueError:continue
            local=[Path(p)/relative for p in row.get("local",[])]
    for path in (new, paid, *local, drive):
        if path is not None and (path.exists() or path.with_name(path.name+".part").exists()):
            return path
    return new


def restore(root, target, url):
    from .transport import cached
    new, drive, paid = locations(root, target)
    selected = locate(root, target)
    if drive is None or selected != drive:
        return selected
    if not cached(url, drive):
        return new
    paid.parent.mkdir(parents=True, exist_ok=True)
    part = paid.with_name(paid.name+".part")
    with drive.open("rb") as src, part.open("xb") as dst:
        shutil.copyfileobj(src, dst, 1024**2)
    part.replace(paid)
    shutil.copyfile(drive.with_name(drive.name+".complete.json"), paid.with_name(paid.name+".complete.json"))
    append(Path(root)/"cache_restore.jsonl",dict(source=str(drive),destination=str(paid),
        copied_bytes=paid.stat().st_size,download_charge_bytes=0,previously_paid=True))
    return paid


def wheels(root, target):
    new, drive, paid = locations(root, target)
    selected = locate(root, target)
    if drive is not None and selected == drive:
        # Lock and successful historical installation were checked by admission.
        shutil.copytree(drive, paid)
        append(Path(root)/"cache_restore.jsonl",dict(source=str(drive),destination=str(paid),
            copied_bytes=sum(p.stat().st_size for p in paid.rglob("*") if p.is_file()),
            download_charge_bytes=0,previously_paid=True))
        return paid
    return selected
