"""Fetch the same upstream CPython build explicitly, without uv's implicit retries."""
import json
import os
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

from qualification.records import append_json, total_bytes

ASSET_API = "https://api.github.com/repos/astral-sh/python-build-standalone/releases/assets/185928887"
ASSET_NAME = "cpython-3.10.14+20240814-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz"
RELEASE = "https://github.com/astral-sh/python-build-standalone/releases/tag/20240814"


def acquire(scratch, receipts, remaining_bytes):
    scratch, receipts = Path(scratch), Path(receipts)
    downloads = scratch/"downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    archive = downloads/ASSET_NAME
    destination = scratch/"python"
    # Every entry is single-use. Partial files are evidence, never a retry cache.
    if archive.exists() or destination.exists():
        raise RuntimeError("Existing Python download/extraction retained; no retry")
    record = dict(asset_api=ASSET_API, release=RELEASE, asset_name=ASSET_NAME,
                  started_unix=time.time(), status="STARTED", retries=0)
    append_json(receipts/"python_download.jsonl", record)
    try:
        # A fresh API request obtains an official redirect, not a cached expiring
        # signed URL. This is one request with redirects, not fallback/retry logic.
        request = urllib.request.Request(ASSET_API+f"?download=1&request={time.time_ns()}",
            headers={"Accept": "application/octet-stream", "X-GitHub-Api-Version": "2022-11-28",
                     "User-Agent": "Model-WM-environment-recovery", "Cache-Control": "no-cache"})
        with urllib.request.urlopen(request, timeout=60) as response, archive.open("xb") as stream:
            length = response.headers.get("Content-Length")
            if length and total_bytes(downloads)+int(length) > remaining_bytes:
                raise RuntimeError("Python payload exceeds remaining download budget")
            while True:
                chunk = response.read(1024*1024)
                if not chunk:
                    break
                stream.write(chunk)
                stream.flush()
                if total_bytes(downloads) > remaining_bytes:
                    raise RuntimeError("Python payload exceeds remaining download budget")
            os.fsync(stream.fileno())
        if length and archive.stat().st_size != int(length):
            raise RuntimeError("Incomplete Python archive; partial payload retained")
        # Validate archive semantics and paths, not a new SHA/size admission gate.
        with tarfile.open(archive, "r:gz") as bundle:
            members = bundle.getmembers()
            base = destination.resolve()
            for member in members:
                target = (base/member.name).resolve()
                if not target.is_relative_to(base) or not member.name.startswith("python/"):
                    raise RuntimeError("Unsafe Python archive member")
                if not (member.isfile() or member.isdir() or member.issym() or member.islnk()):
                    raise RuntimeError("Unsupported Python archive member")
                if member.issym() or member.islnk():
                    link = (target.parent/member.linkname if member.issym() else base/member.linkname).resolve()
                    if not link.is_relative_to(base):
                        raise RuntimeError("Unsafe Python archive link")
            destination.mkdir()
            # Extraction order cannot redirect a later member outside base: all
            # links have contained resolved targets and extraction is into a new dir.
            bundle.extractall(destination, members=members, filter="data")
        python = destination/"python/bin/python3.10"
        version = subprocess.check_output([str(python), "--version"], text=True).strip()
        if version != "Python 3.10.14":
            raise RuntimeError(f"Unexpected interpreter version: {version}")
        append_json(receipts/"python_download.jsonl", dict(record, status="COMPLETED",
                    bytes=archive.stat().st_size, version=version, finished_unix=time.time()))
        print(json.dumps(dict(python=str(python), version=version, bytes=archive.stat().st_size)))
        return python
    except BaseException as exc:
        append_json(receipts/"python_download.jsonl", dict(record, status="FAILED",
                    bytes=archive.stat().st_size if archive.exists() else 0,
                    error=f"{type(exc).__name__}: {exc}", finished_unix=time.time()))
        raise


if __name__ == "__main__":
    acquire(sys.argv[1], sys.argv[2], int(sys.argv[3]))
