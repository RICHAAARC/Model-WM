"""Actual official LaWa graph/state names, only random CPU-generated tensors."""
from pathlib import Path
import tempfile,shutil,json,hashlib
from unittest.mock import patch
import numpy as np
import torch
from .model import Backend
from lawa_ab.model import build_writer
from matched_loss.carrier import Carrier

def check(source):
    torch.set_num_threads(1);torch.manual_seed(25)
    package=Path(__file__).parent
    with tempfile.TemporaryDirectory() as tmp:
        s=Path(tmp);(s/'official').symlink_to(source.resolve(),target_is_directory=True)
        shutil.copytree(package/'source_evidence/sd21',s/'cache/sd21')
        shutil.copytree(package/'source_evidence/sd14',s/'cache/sd14')
        w=build_writer(source)
        shapes={k:list(v.shape) for k,v in w.state_dict().items()}
        expected=json.loads((package.parent/'lawa_ab/schema.json').read_text())
        assert len(shapes)==expected['keys']
        assert hashlib.sha256(json.dumps(shapes,sort_keys=True).encode()).hexdigest()==expected['keys_shapes_sha256']
        torch.save({'state_dict':w.state_dict()},s/'cache/last.ckpt');del w
        # The real tiny Diffusers pipeline is covered separately; this check
        # isolates strict production Backend.load + official writer/reader.
        with patch('sd2_matched.model.build_pipeline',return_value=object()):
            b=Backend.load(s,'cpu')
        raw=torch.randn(1,4,8,8)
        h=b.writer.ae.post_quant_conv(raw)
        c=Carrier(b,.1);bits='01'*24
        native=b.write(h,bits);ordinary=c.uniform(h,bits,1.025);coarse=c.write(h,bits,'coarse')
        for image in (native,ordinary,coarse):
            assert image.shape==(64,64,3) and image.dtype==np.uint8
            assert len(b.read(image))==48
        assert all(not conv._forward_hooks for _,conv,_ in c.branches())
        assert np.array_equal(native,b.write(h,bits))
        print(json.dumps(dict(official_state_keys=len(shapes),strict_synthetic_load=True,
             official_writer_reader=True,three_actions=True,hook_restoration=True,
             real_weights_loaded=False,gpu_started=False)))
if __name__=='__main__':
    import sys
    check(Path(sys.argv[1]))
