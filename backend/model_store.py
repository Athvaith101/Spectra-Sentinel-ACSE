from __future__ import annotations

import os
from pathlib import Path


def ensure_model_artifacts(root: Path) -> Path:
    """Ensure deployment artifacts exist locally.

    Local development uses bundled artifacts. Production can set
    MODEL_SOURCE=huggingface and HF_MODEL_REPO to hydrate the same
    ``models/frozen`` and ``models/rl`` directories at startup.
    """
    model_root = root / "models"
    stare = model_root / "frozen" / "stare_inference.pt"
    source = os.environ.get("MODEL_SOURCE", "local").strip().lower()
    if source != "huggingface" or stare.exists():
        return model_root

    repo = os.environ.get("HF_MODEL_REPO", "").strip()
    if not repo:
        raise RuntimeError("MODEL_SOURCE=huggingface requires HF_MODEL_REPO")

    from huggingface_hub import snapshot_download

    revision = os.environ.get("HF_MODEL_REVISION", "main")
    token = os.environ.get("HF_TOKEN") or None
    model_root.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=repo,
        revision=revision,
        local_dir=str(model_root),
        token=token,
        allow_patterns=["frozen/*", "rl/*", "models/frozen/*", "models/rl/*", "README.md"],
    )

    # Accept either a repository whose root is frozen/rl or one whose root is models/frozen/rl.
    nested = model_root / "models"
    if nested.is_dir():
        for child in nested.iterdir():
            target = model_root / child.name
            if not target.exists():
                child.replace(target)

    if not stare.exists():
        raise RuntimeError(
            f"Hugging Face repository {repo} did not provide frozen/stare_inference.pt"
        )
    return model_root
