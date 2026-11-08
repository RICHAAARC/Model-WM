"""One-shot public asset downloads. No model construction or inference."""
import json
import os
import sys
import urllib.request
from pathlib import Path
from .records import append_json, total_bytes


def fetch(url, target, root, download_root, limit):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise RuntimeError(f"Unexpected preexisting asset: {target.name}")
    # Retries are intentionally absent. Files left partial remain as failure evidence.
    with urllib.request.urlopen(url, timeout=60) as response:
        length = response.headers.get("Content-Length")
        used = total_bytes(download_root)+256*1024**2
        if length and used+int(length) > limit:
            raise RuntimeError("Projected asset download exceeds 10 GiB including setup reserve")
        received = 0
        with target.open("xb") as out:
            while True:
                chunk = response.read(1024*1024)
                if not chunk:
                    break
                if used+received+len(chunk) > limit:
                    raise RuntimeError("Download budget reached")
                out.write(chunk)
                received += len(chunk)
        if length and received != int(length):
            raise RuntimeError("Incomplete download; no automatic retry")
    append_json(root/"downloads.jsonl", dict(url=url, relative_path=str(target.relative_to(download_root)), bytes=received))


def main():
    root, download_root = map(Path, sys.argv[1:3])
    manifest = json.loads((root/"manifest.json").read_text())
    assets = download_root/"assets"
    base_files = ["model_index.json", "scheduler/scheduler_config.json", "text_encoder/config.json",
                  "text_encoder/model.safetensors", "tokenizer/merges.txt", "tokenizer/special_tokens_map.json",
                  "tokenizer/tokenizer_config.json", "tokenizer/vocab.json", "unet/config.json",
                  "unet/diffusion_pytorch_model.safetensors", "vae/config.json", "vae/diffusion_pytorch_model.safetensors"]
    planned = [(f"https://huggingface.co/{manifest['base_repo']}/resolve/{manifest['base_revision']}/{f}", assets/"base"/f) for f in base_files]
    planned += [(f"https://huggingface.co/{manifest['model_repo']}/resolve/{manifest['model_revision']}/ppft_trained/{f}", assets/"aqualora"/f)
                for f in ["mapper.pt", "msgdecoder.pt", "pytorch_lora_weights.safetensors"]]
    planned += [(f"https://raw.githubusercontent.com/Georgefwt/AquaLoRA/{manifest['source_revision']}/{f}", assets/"official"/f)
                for f in ["utils/models.py", "scripts/create_wm_lora.py", "requirements.txt"]]
    for url, target in planned:
        fetch(url, target, root, download_root, manifest["budget"]["download_bytes"])
        print(f"Downloaded {target.relative_to(assets)}", flush=True)


if __name__ == "__main__":
    main()
