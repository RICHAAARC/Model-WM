"""Reuse already-paid software wheels without hiding new downloads."""
import email
import re
import zipfile
from pathlib import Path


def wheel_identity(path):
    with zipfile.ZipFile(path) as wheel:
        metadata = [n for n in wheel.namelist() if n.endswith(".dist-info/METADATA")]
        if len(metadata) != 1:
            raise RuntimeError("Unreadable wheel metadata; cache source unresolved")
        info = email.message_from_bytes(wheel.read(metadata[0]))
    return re.sub(r"[-_.]+", "-", info["Name"]).lower(), info["Version"]


def paid_links(source, destination, expected):
    """Only the successful recovery's own cache is supplied by the caller."""
    result = {}
    for path in sorted(Path(source).glob("*.whl")):
        name, version = wheel_identity(path)
        if expected.get(name) != version or path.is_symlink():
            raise RuntimeError(f"Unknown cached wheel source/version: {path.name}")
        target = Path(destination)/path.name
        target.symlink_to(path.resolve())
        result[str(target)] = dict(
            source=str(path.resolve()), bytes=path.stat().st_size)
    return result


def charged_bytes(download_root, prepaid=None):
    prepaid = prepaid or {}
    total = 0
    for path in Path(download_root).rglob("*"):
        record = prepaid.get(str(path))
        if record:
            if not path.is_symlink() or str(path.resolve()) != record["source"] or path.stat().st_size != record["bytes"]:
                raise RuntimeError("Previously paid cache link changed; accounting unresolved")
        elif path.is_file():
            total += path.stat().st_size
    return total
