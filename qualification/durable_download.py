"""Persistent official-source downloads; transport recovery is not a trajectory."""
import json
import os
import time
import urllib.request
from pathlib import Path

from .records import append_json, write_json


def fetch(url, target, log, *, headers=None, attempts=3, opener=urllib.request.urlopen):
    target, log = Path(target), Path(log)
    target.parent.mkdir(parents=True, exist_ok=True)
    complete = target.with_name(target.name+".complete.json")
    if target.exists() and complete.exists():
        record = json.loads(complete.read_text())
        if record.get("url") != url:
            raise ValueError("Cache source differs from the requested official artifact")
        if target.stat().st_size == record.get("bytes"):
            return target
    part = target.with_name(target.name+".part")
    for attempt in range(attempts):
        start = part.stat().st_size if part.exists() else 0
        received = 0
        try:
            request_headers = dict(headers or {})
            if start:
                request_headers["Range"] = f"bytes={start}-"
            # A fresh API redirect avoids an expired cached signed Python URL.
            request_url = url
            if "api.github.com/repos/astral-sh/" in url:
                request_url += f"?download=1&request={time.time_ns()}"
            request = urllib.request.Request(request_url, headers=request_headers)
            with opener(request, timeout=60) as response:
                status = getattr(response, "status", response.getcode())
                append = start > 0 and status == 206
                if append and not response.headers.get("Content-Range", "").startswith(f"bytes {start}-"):
                    raise ValueError("Server returned a different byte range")
                if not append:
                    start = 0  # Range ignored: log the extra transfer, never concatenate.
                length = response.headers.get("Content-Length")
                with part.open("ab" if append else "wb") as out:
                    while True:
                        block = response.read(4*1024**2)
                        if not block:
                            break
                        out.write(block)
                        received += len(block)
                    out.flush()
                    os.fsync(out.fileno())
                if length and received != int(length):
                    raise OSError("Incomplete HTTP body; partial cache retained")
            part.replace(target)
            write_json(complete, dict(url=url, bytes=target.stat().st_size))
            append_json(log, dict(url=url, status="COMPLETED", attempt=attempt+1,
                                 resumed_from=start, received_bytes=received))
            print(f"Cached {target.name}: {target.stat().st_size/1024**2:.1f} MiB", flush=True)
            return target
        except Exception as exc:
            append_json(log, dict(url=url, status="TRANSPORT_FAILED", attempt=attempt+1,
                resumed_from=start, received_bytes=received, error=f"{type(exc).__name__}: {exc}"))
            if attempt+1 == attempts:
                raise
            time.sleep(min(2**attempt, 4))


def main():
    import sys
    from .prepare import asset_plan
    root, cache = map(Path, sys.argv[1:3])
    manifest = json.loads((root/"manifest.json").read_text())
    for url, target in asset_plan(manifest, cache):
        fetch(url, target, root/"download_events.jsonl")


if __name__ == "__main__":
    main()
