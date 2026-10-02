from __future__ import annotations

import os
from pathlib import Path
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
repo_id = os.environ.get("HF_MODEL_REPO")
if not repo_id:
    raise SystemExit("Set HF_MODEL_REPO first, e.g. Athvaith/sih26055-ew-final-model")

folder = ROOT / "models"
if not (folder / "frozen" / "stare_inference.pt").exists():
    raise SystemExit("models/frozen/stare_inference.pt is missing")

api = HfApi()
api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
url = api.upload_folder(
    repo_id=repo_id,
    repo_type="model",
    folder_path=str(folder),
    path_in_repo=".",
    commit_message="SIH26055 v6.2 frozen deployment artifacts",
)
print(url)
