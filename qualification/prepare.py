"""One-shot public asset downloads. No model construction or inference."""
import json
import os
import sys
import urllib.request
from pathlib import Path
from .records import append_json, total_bytes, write_json
from .runtime_cache import charged_bytes


def fetch(url, target, root, download_root, limit, reserve=256*1024**2, prepaid=None):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise RuntimeError(f"Unexpected preexisting asset: {target.name}")
    # Retries are intentionally absent. Files left partial remain as failure evidence.
    with urllib.request.urlopen(url, timeout=60) as response:
        length = response.headers.get("Content-Length")
        used = charged_bytes(download_root, prepaid)+reserve
        if length and used+int(length) > limit:
            raise RuntimeError("Projected asset download exceeds remaining cumulative budget")
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


def asset_plan(manifest, download_root):
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
    return planned


def preflight(root, download_root, planned, limit, reserve, prepaid):
    rows = []
    for url, target in planned:
        if target.exists():
            raise RuntimeError("Unregistered model cache exists; source unresolved")
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=30) as response:
            size = response.headers.get("Content-Length")
        if size is None or not size.isdigit() or int(size) <= 0:
            raise RuntimeError(f"Asset size unavailable; projected cost unresolved: {target.name}")
        rows.append(dict(url=url, bytes=int(size), relative_path=str(target.relative_to(download_root))))
    environment = json.loads((root/"environment_plan.json").read_text())
    projected = charged_bytes(download_root, prepaid)+reserve+sum(r["bytes"] for r in rows)+environment["estimated_new_software_bytes"]
    write_json(root/"download_preflight.json", dict(assets=rows, projected_segment_bytes=projected,
        segment_limit_bytes=limit, software_estimate=environment["estimated_new_software_bytes"],
        status="PROJECTED_DOWNLOAD_BUDGET_EXCEEDED" if projected > limit else "WITHIN_DOWNLOAD_ESTIMATE"))
    if projected > limit:
        raise RuntimeError("Projected software plus model downloads exceed remaining original budget")


def main():
    root, download_root = map(Path, sys.argv[1:3])
    manifest = json.loads((root/"manifest.json").read_text())
    planned = asset_plan(manifest, download_root)
    context = root/"download_budget.json"
    if context.exists():
        budget = json.loads(context.read_text())
        limit = manifest["budget"]["download_bytes"]-512*1024**2-budget["carried_download_bytes"]
        reserve, prepaid = 0, budget["prepaid"]
    else:
        limit, reserve, prepaid = manifest["budget"]["download_bytes"], 256*1024**2, {}
    if "--preflight" in sys.argv[3:]:
        preflight(root, download_root, planned, limit, reserve, prepaid)
        return
    for url, target in planned:
        fetch(url, target, root, download_root, limit, reserve, prepaid)
        print(f"Downloaded {target.relative_to(download_root/'assets')}", flush=True)


if __name__ == "__main__":
    main()
