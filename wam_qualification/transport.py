"""Official publisher transport repair; no TLS bypass or fallback retries."""
import json
import urllib.request
from urllib.parse import urlsplit
from pathlib import Path
from .core import append, save

ORIGINAL = "https://dl.fbaipublicfiles.com/watermark_anything/wam_mit.pth"
HF_REVISION = "88ffb2cd1079a76b01d6beda8bf2a433c25fa881"
OFFICIAL_MIRROR = f"https://huggingface.co/facebook/watermark-anything/resolve/{HF_REVISION}/checkpoint.pth"


def transport_url(url):
    return OFFICIAL_MIRROR if url == ORIGINAL else url


def cached(url, target):
    target = Path(target)
    if not target.exists():
        if target.with_name(target.name+".part").exists():
            raise RuntimeError("Partial transport retained; no retry")
        return False
    receipt = target.with_name(target.name+".complete.json")
    if not receipt.is_file() or json.loads(receipt.read_text()).get("url") != url:
        raise RuntimeError("Cached asset source is unresolved; no replacement")
    if not target.stat().st_size:
        raise RuntimeError("Empty cached artifact")
    return True


def head(url):
    resolved = transport_url(url)
    request = urllib.request.Request(resolved, method="HEAD", headers={"User-Agent":"Model-WM-WAM"})
    with urllib.request.urlopen(request, timeout=20) as response:
        if response.status != 200 or urlsplit(response.url).scheme != "https":
            raise RuntimeError("Official asset preflight requires HTTPS HTTP 200")
        return dict(url=url, transport_url=resolved, status=response.status,
            final_host=urlsplit(response.url).hostname, reported_bytes=response.headers.get("Content-Length"),
            body_downloaded=False, certificate_verification=True)


def preflight(root, oldcache, cache):
    from .setup import BASE_FILES
    root, oldcache, cache = map(Path, (root, oldcache, cache))
    m = json.loads((root/"manifest.json").read_text())
    targets = [(f"https://codeload.github.com/facebookresearch/watermark-anything/tar.gz/{m['source_commit']}", cache/"official_source.tar.gz"),
               (m["checkpoint_url"], cache/"wam_mit.pth")]
    for name in BASE_FILES:
        url = f"https://huggingface.co/{m['base_repo']}/resolve/{m['base_revision']}/{name}"
        paid = oldcache/"assets/base"/name
        targets.append((url, paid if paid.exists() else cache/"base"/name))
    for url, path in targets:
        try:
            row = dict(url=url, status="PAID_CACHE", body_downloaded=False) if cached(url,path) else head(url)
            append(root/"asset_preflight.jsonl", row)
        except Exception as exc:
            append(root/"asset_preflight.jsonl", dict(url=url, status="FAILED", error=str(exc), body_downloaded=False))
            raise
    save(root/"asset_preflight.json", dict(status="HEAD_AND_CACHE_CHECKS_PASSED", count=len(targets),
        weights_loaded=False, note="HEAD availability is not checkpoint validity or model qualification"))


if __name__ == "__main__":
    import sys
    preflight(*sys.argv[1:4])
