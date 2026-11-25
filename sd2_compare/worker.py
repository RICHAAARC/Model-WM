"""Use the unchanged, paired CAL/seal/TEST executor with the audited SD2 backend."""
import json,sys
from pathlib import Path
from matched_compare.worker import run
from sd2_matched.model import Backend
from matched_loss.carrier import Carrier
from latent_control.io import save
PACKAGE=Path(__file__).resolve().parent

if __name__=='__main__':
    local,out,scratch=map(Path,sys.argv[1:])
    backend=Backend.load(scratch)
    save(local/'outcome.json',run(local,out,json.loads((PACKAGE/'allocation.json').read_text()),backend,Carrier(backend,.1)))
