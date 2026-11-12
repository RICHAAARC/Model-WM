"""Linux stage owner: reap orphan descendants and seal CPU after cancellation."""
import ctypes
import json
import os
import resource
import signal
import subprocess
import sys
import time
from pathlib import Path
import psutil
from .core import save


def supervise(receipt, command):
    receipt=Path(receipt)
    if ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0):  # PR_SET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(), "Cannot account for orphan descendants")
    started=time.monotonic()
    stop=[False]
    def request_stop(signum, frame): stop[0]=True
    signal.signal(signal.SIGTERM,request_stop)
    signal.signal(signal.SIGINT,request_stop)
    process=subprocess.Popen(command,start_new_session=True)
    reaped_cpu=0.; observed=0.; exit_code=None; stopping=None; peak_ram=0
    while True:
        # Every reaped child's rusage includes only descendants it already reaped.
        # Orphans adopted here are separate children, so their wait4 usage is additive.
        no_children=False
        while True:
            try: pid,status,usage=os.wait4(-1,os.WNOHANG)
            except ChildProcessError:
                no_children=True;break
            if pid==0: break
            reaped_cpu+=usage.ru_utime+usage.ru_stime
            if pid==process.pid: exit_code=os.waitstatus_to_exitcode(status)
        own=resource.getrusage(resource.RUSAGE_SELF)
        own_cpu=own.ru_utime+own.ru_stime
        live_cpu=0.; ram=psutil.Process().memory_info().rss
        descendants=psutil.Process().children(recursive=True)
        for p in descendants:
            try:
                t=p.cpu_times();live_cpu+=t.user+t.system+t.children_user+t.children_system
                ram+=p.memory_info().rss
            except psutil.NoSuchProcess: pass
        observed=max(observed,own_cpu+reaped_cpu+live_cpu)
        peak_ram=max(peak_ram,ram)
        finished=no_children
        save(receipt,dict(cpu_seconds=observed,wall_seconds=time.monotonic()-started,
            peak_ram_bytes=peak_ram,resources_final=finished,exit_code=exit_code,
            interrupted=stop[0],all_descendants_reaped=finished,
            measurement="max(observed tree CPU, owner plus wait4 child subtree CPU)"))
        if finished:
            process.returncode=exit_code
            return 130 if stop[0] else (exit_code or 0)
        if stop[0] or exit_code is not None:
            if stopping is None: stopping=time.monotonic()
            sig=signal.SIGKILL if time.monotonic()-stopping>2 else signal.SIGTERM
            # Include adopted descendants and separate process groups; keep this
            # accounting owner alive until every child has been reaped.
            for p in descendants:
                try: p.send_signal(sig)
                except psutil.NoSuchProcess: pass
        time.sleep(.05)


if __name__=="__main__":sys.exit(supervise(sys.argv[1],sys.argv[2:]))
