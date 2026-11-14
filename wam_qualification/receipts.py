"""Local IPC and quiescent publication; Drive is never a live mailbox."""
import json
import shutil
from pathlib import Path


def read_receipt(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Cannot read resource receipt {path}: {exc}") from exc


def publish_tree(source, destination, exclude=()):
    """Called only after writers stop. Existing historical invocations are untouched."""
    source, destination = Path(source), Path(destination)
    if source == destination:
        return
    for src in sorted(source.rglob("*")):
        relative = src.relative_to(source)
        if not src.is_file() or src.is_symlink() or any(p in exclude for p in relative.parts):
            continue
        dst = destination / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and dst.read_bytes() == src.read_bytes():
            continue
        temp = dst.with_name(dst.name + ".publishing")
        shutil.copyfile(src, temp)
        temp.replace(dst)


def tree_bytes(root):
    # active.lock is removed after sealing; it is not part of the final result.
    return sum(p.stat().st_size for p in Path(root).rglob("*")
               if p.is_file() and not p.is_symlink() and p != Path(root)/"active.lock")
