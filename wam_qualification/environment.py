"""Import and exact version check. Does not construct a model or read weights."""
import importlib.metadata
import json
import sys
from pathlib import Path


def check(cuda=False):
    if sys.version_info[:3] != (3, 10, 14):
        raise RuntimeError("Expected official-tested Python 3.10.14")
    lock = Path(__file__).with_name("requirements-colab.txt" if cuda else "requirements-cpu.txt")
    versions = {}
    for line in lock.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        name, expected = line.split("==")
        got = importlib.metadata.version(name)
        if got != expected:
            raise RuntimeError(f"Version mismatch {name}: {got} != {expected}")
        versions[name] = got
    import torch, torchvision, diffusers, transformers, omegaconf, timm, cv2, skimage
    if cuda and (torch.version.cuda != "12.4" or not torch.cuda.is_available()):
        raise RuntimeError("Pinned CUDA runtime unavailable")
    return dict(python=sys.version, versions=versions, cuda=torch.version.cuda, real_weights_loaded=False)


if __name__ == "__main__":
    print(json.dumps(check("--cuda" in sys.argv), indent=2))
