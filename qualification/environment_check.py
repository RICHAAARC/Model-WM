"""Check installed package metadata only; never import torch or load a model."""
import importlib.metadata
import json
import platform
import re
import sys
from pathlib import Path


def pins(text):
    result = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.count("==") != 1:
            raise ValueError("Expected an exact package version")
        name, version = line.split("==")
        result[re.sub(r"[-_.]+", "-", name).lower()] = version
    return result


def check(lock):
    if platform.python_version() != "3.10.14":
        raise RuntimeError("Expected Python 3.10.14")
    installed = {}
    for name, version in pins(Path(lock).read_text()).items():
        installed[name] = importlib.metadata.version(name)
        if installed[name] != version:
            raise RuntimeError(f"Installed version differs: {name} {installed[name]} != {version}")
    print(json.dumps(dict(python=platform.python_version(), packages=installed), sort_keys=True))


if __name__ == "__main__":
    check(sys.argv[1])
