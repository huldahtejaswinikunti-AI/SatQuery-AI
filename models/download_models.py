import argparse
from pathlib import Path

MODELS = {
    "clipseg": "CIDAS/clipseg-rd64",
    "geochat": "MBZUAI/geochat-7B",
    "tinycd": "AndreaCodegoni/Tiny_model_4_CD",
    "phrasing": "microsoft/Phi-3-mini-4k-instruct"
}

def download_models(target_dir: str = "models"):
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Install huggingface_hub: pip install huggingface_hub")
        return
    base = Path(target_dir)
    for name, repo in MODELS.items():
        dest = base / name
        print(f"Downloading {repo} to {dest}...")
        try:
            snapshot_download(repo_id=repo, local_dir=str(dest), resume_download=True)
        except Exception as e:
            print(f"Failed {repo}: {e}")

if __name__ == "__main__":
    download_models()
