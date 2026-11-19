"""Validate archived bytes and execute every published cell with a synthetic Colab mount."""
import ast
import base64
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType,SimpleNamespace
from unittest.mock import patch
from zipfile import ZipFile
import nbformat

def main():
    repo=Path(__file__).resolve().parents[1]
    notebook=nbformat.read(repo/'notebooks/terminal_solver_colab.ipynb',as_version=4)
    nbformat.validate(notebook)
    cells=[c.source for c in notebook.cells if c.cell_type=='code']
    assert len(cells)==3 and cells[0].strip()=="from google.colab import drive\ndrive.mount('/content/drive')"
    calls=[n for n in ast.walk(ast.parse(cells[1])) if isinstance(n,ast.Call) and
           isinstance(n.func,ast.Attribute) and n.func.attr=='b64decode']
    with ZipFile(io.BytesIO(base64.b64decode(ast.literal_eval(calls[0].args[0])))) as payload:
        for name in payload.namelist():
            assert payload.read(name)==(repo/name).read_bytes(),name
        assert 'terminal_solver/allocation.json' in payload.namelist()
        assert 'wam_qualification/requirements-colab.txt' in payload.namelist()
        print('Embedded source files:',len(payload.namelist()))
    events=[]
    colab=ModuleType('google.colab')
    colab.drive=SimpleNamespace(mount=lambda p:events.append(('mount',p)))
    colab.runtime=SimpleNamespace(unassign=lambda:events.append('unassign'))
    google=ModuleType('google')
    google.colab=colab
    with tempfile.TemporaryDirectory() as td:
        original=tempfile.mkdtemp
        def redirected(*args,**kwargs):
            if kwargs.get('dir')=='/content':
                kwargs['dir']=td
            return original(*args,**kwargs)
        with patch.dict(sys.modules,{'google':google,'google.colab':colab}), patch.dict(os.environ,{'PATH':''}),patch.object(tempfile,'mkdtemp',redirected):
            ns={}
            for code in cells:
                exec(compile(code,'terminal_solver_colab','exec'),ns)
            assert ns['result']['status']=='GPU_REQUIRED' and events==[('mount','/content/drive')]
    print('All notebook cells completed CPU-only, GPU start guard and mount-only first cell verified.')

if __name__=='__main__':
    main()
