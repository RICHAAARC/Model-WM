"""Recover compatible packages in the current runtime; never trust old READY alone."""
import json
import subprocess
import sys
from pathlib import Path
from .core import save
from .setup import RUNTIME_ROOT, PYTHON_API, PYTHON_ASSET, call, fetch, unpack
from .storage import locate, wheels as restore_wheels


def environment(root,scratch,oldcache,cache):
    root,scratch,oldcache,cache=map(Path,(root,scratch,oldcache,cache))
    venv=RUNTIME_ROOT/"wam-pinned-v1/venv"
    python=venv/"bin/python"
    observation=dict(runtime_boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        python_exists_now=python.is_file(),previous_ready_receipt_is_not_proof=True)
    save(root/"environment_observation.json",observation)
    if python.is_file():
        probe=subprocess.run([str(python),"-m","wam_qualification.environment","--cuda"])
        if probe.returncode==0:
            save(root/"python.json",dict(python=str(python),reused=True,validated_in_current_runtime=True))
            return
        # A known unfinished installation may be repaired from its pinned wheels.
        # A working interpreter must still have the exact frozen Python version.
        call([python,"-c","import sys; assert sys.version_info[:3] == (3,10,14), sys.version"])
    else:
        target=oldcache/PYTHON_ASSET if locate(root,oldcache/PYTHON_ASSET).is_file() else cache/PYTHON_ASSET
        archive=fetch(PYTHON_API,target,root,{"Accept":"application/octet-stream","User-Agent":"Model-WM-WAM"})
        unpack(archive,scratch/"python")
        base=scratch/"python/python/bin/python3.10"
        call([base,"-c","import sys; assert sys.version_info[:3] == (3,10,14), sys.version"])
        # venv creation retains installed site-packages if a prior creation was
        # interrupted. A broken old base interpreter is replaced from paid source.
        call([base,"-m","venv",venv])
    wheels=restore_wheels(root,cache/"wheels")
    lock=Path(__file__).with_name("requirements-colab.txt")
    if not wheels.exists():
        wheels.mkdir(parents=True)
        call([python,"-m","pip","download","--no-cache-dir","--retries","0","--dest",wheels,
            "--extra-index-url","https://download.pytorch.org/whl/cu124","-r",lock])
    # pip retains already satisfied exact versions; it installs missing packages.
    call([python,"-m","pip","install","--no-index","--find-links",wheels,"-r",lock])
    call([python,"-m","pip","check"])
    call([python,"-m","wam_qualification.environment","--cuda"])
    save(root/"python.json",dict(python=str(python),reused=False,
        repaired_existing_interpreter=observation["python_exists_now"],validated_in_current_runtime=True))


if __name__=="__main__":environment(*sys.argv[1:5])
