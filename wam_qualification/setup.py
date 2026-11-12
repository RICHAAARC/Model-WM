"""Single transport attempt, official sources, isolated pinned software."""
import json
import os
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path
from .core import append, save
from .transport import cached, transport_url
from .storage import locate, restore, wheels as restore_wheels

RUNTIME_ROOT = Path("/content/model-wm-runtime")

PYTHON_ASSET = "cpython-3.10.14+20240814-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz"
PYTHON_API = "https://api.github.com/repos/astral-sh/python-build-standalone/releases/assets/185928887"
BASE_FILES = ["model_index.json", "scheduler/scheduler_config.json", "text_encoder/config.json",
    "text_encoder/model.safetensors", "tokenizer/merges.txt", "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer_config.json", "tokenizer/vocab.json", "unet/config.json",
    "unet/diffusion_pytorch_model.safetensors", "vae/config.json", "vae/diffusion_pytorch_model.safetensors"]


def unpack(archive, target, strip=False, source_only=False):
    target = Path(target).resolve()
    target.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive) as tf:
        for member in tf.getmembers():
            name = Path(*Path(member.name).parts[1:]) if strip else Path(member.name)
            if str(name) == ".":
                continue
            dst = (target/name).resolve()
            if target not in dst.parents:
                raise RuntimeError("Unsafe archive path")
            if source_only and dst.suffix not in (".py", ".json", ".yaml", ".txt", ".md") and dst.name != "LICENSE":
                continue
            if member.isdir():
                dst.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                dst.parent.mkdir(parents=True, exist_ok=True)
                with tf.extractfile(member) as src, dst.open("xb") as out:
                    while block := src.read(1024**2):
                        out.write(block)
                dst.chmod(member.mode & 0o777)
            elif member.issym() and not source_only:
                link = (dst.parent/member.linkname).resolve()
                if target not in link.parents:
                    raise RuntimeError("Unsafe archive link")
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.symlink_to(member.linkname)
            elif not source_only:
                raise RuntimeError("Unsupported archive member")


def fetch(url, target, root, headers=None):
    target = restore(root, Path(target), url)
    target.parent.mkdir(parents=True, exist_ok=True)
    receipt = target.with_name(target.name+".complete.json")
    if cached(url, target):
        append(root/"downloads.jsonl", dict(url=url, path=str(target), status="REUSED", received_bytes=0))
        return target
    part = target.with_name(target.name+".part")
    received = 0
    try:
        request = urllib.request.Request(transport_url(url), headers=headers or {})
        with urllib.request.urlopen(request, timeout=30) as response, part.open("xb") as out:
            length = response.headers.get("Content-Length")
            while block := response.read(1024**2):
                out.write(block); received += len(block)
            out.flush(); os.fsync(out.fileno())
        if length and received != int(length):
            raise RuntimeError("Truncated HTTP response")
        part.replace(target)
        save(receipt, dict(url=url, transport_url=transport_url(url), bytes=received))
        append(root/"downloads.jsonl", dict(url=url, transport_url=transport_url(url), path=str(target), status="COMPLETED", received_bytes=received))
        return target
    except BaseException as exc:
        append(root/"downloads.jsonl", dict(url=url, transport_url=transport_url(url), path=str(part), status="FAILED", received_bytes=received, error=str(exc)))
        raise


def call(args):
    print(" ".join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), check=True)


def environment(root, scratch, oldcache, cache):
    from .environment import check
    # An already compatible environment can be reused; the old Aqua environment
    # has different torch pins and must not be mutated to satisfy WAM.
    reusable = RUNTIME_ROOT/"wam-pinned-v1/venv/bin/python"
    if reusable.is_file():
        call([reusable, "-m", "wam_qualification.environment", "--cuda"])
        save(root/"python.json", dict(python=str(reusable), reused=True))
        return
    oldpy = RUNTIME_ROOT/"aqualora-qualification-20261108-attempt01/persistent-runtime/python/python/bin/python3.10"
    if oldpy.is_file():
        py = oldpy
    else:
        paid = locate(root, oldcache/PYTHON_ASSET)
        if paid.is_file():
            receipt = paid.with_name(paid.name+".complete.json")
            if not receipt.is_file() or json.loads(receipt.read_text()).get("url") != PYTHON_API:
                raise RuntimeError("Python cache source is unresolved")
            archive = fetch(PYTHON_API, oldcache/PYTHON_ASSET, root)
        else:
            archive = fetch(PYTHON_API, cache/PYTHON_ASSET, root,
                {"Accept":"application/octet-stream", "User-Agent":"Model-WM-WAM", "Cache-Control":"no-cache"})
        unpack(archive, scratch/"python")
        py = scratch/"python/python/bin/python3.10"
    call([py, "-c", "import sys; assert sys.version_info[:3] == (3,10,14), sys.version"])
    venv = RUNTIME_ROOT/"wam-pinned-v1/venv"
    if venv.exists():
        raise RuntimeError("Partial existing environment retained; no repair/retry")
    call([py, "-m", "venv", venv])
    python = venv/"bin/python"
    wheels = restore_wheels(root, cache/"wheels")
    lock = Path(__file__).with_name("requirements-colab.txt")
    if wheels.exists():
        # Recovery admission verifies the original successful environment receipt
        # and the exact frozen lock. Missing cached packages fail offline; no retry.
        if not (root/"recovery_admission.json").is_file():
            raise RuntimeError("Existing wheels require an admitted paid-cache recovery")
    else:
        wheels.mkdir()
        call([python, "-m", "pip", "download", "--no-cache-dir", "--retries", "0", "--dest", wheels,
            "--extra-index-url", "https://download.pytorch.org/whl/cu124", "-r", lock])
    call([python, "-m", "pip", "install", "--no-index", "--find-links", wheels, "-r", lock])
    call([python, "-m", "pip", "check"])
    call([python, "-m", "wam_qualification.environment", "--cuda"])
    save(root/"python.json", dict(python=str(python), reused=False))


def models(root, scratch, oldcache, cache):
    m = json.loads((root/"manifest.json").read_text())
    source_url = f"https://codeload.github.com/facebookresearch/watermark-anything/tar.gz/{m['source_commit']}"
    archive = fetch(source_url, cache/"official_source.tar.gz", root)
    source = scratch/"official_source"
    unpack(archive, source, strip=True, source_only=True)
    params = json.loads((source/"checkpoints/params.json").read_text())
    expected = dict(embedder_model="vae_small", extractor_model="sam_base", nbits=32, img_size=256,
                    attenuation="jnd_1_3_blue", scaling_w=2.0, scaling_i=1.0)
    if any(params.get(k) != v for k,v in expected.items()):
        raise RuntimeError("Official model configuration does not match preregistration")
    # Catch inherited Colab frontend/import incompatibility before any model load.
    sys.path[:0] = [str(source), str(source/"notebooks")]
    import matplotlib
    from inference_utils import load_model_from_checkpoint
    if matplotlib.get_backend().lower() != "agg":
        raise RuntimeError("Headless official imports require Agg")
    save(root/"official_import_preflight.json", dict(status="PASSED", backend=matplotlib.get_backend(),
        weights_loaded=False, physical_operations_started=0))
    checkpoint = fetch(m["checkpoint_url"], cache/"wam_mit.pth", root)
    base = oldcache/"assets/base"
    # Same accepted base revision; a missing paid asset is downloaded once into
    # the new cache. Never silently mix an unknown cached base with this revision.
    paths = []
    for name in BASE_FILES:
        url = f"https://huggingface.co/{m['base_repo']}/resolve/{m['base_revision']}/{name}"
        if locate(root, base/name).exists():
            paths.append((name, fetch(url, base/name, root)))
        else:
            paths.append((name, fetch(url, cache/"base"/name, root)))
    localbase = scratch/"base"
    localbase.mkdir()
    for name, src in paths:
        dest = localbase/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.symlink_to(src.resolve())
    save(root/"model_paths.json", dict(source=str(source), checkpoint=str(checkpoint), base=str(localbase)))


if __name__ == "__main__":
    action = sys.argv[1]
    args = list(map(Path, sys.argv[2:6]))
    {"environment":environment, "models":models}[action](*args)
