import json
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load YAML config."""
    with Path(path).open("r") as f:
        return yaml.safe_load(f)


def save_metadata(metadata: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(metadata, f, indent=2)


def ensure_artifact_dirs(config: Dict[str, Any]) -> None:
    """Create model/report directories if they do not exist."""
    artifacts = config.get("artifacts", {})
    for key in ("model_dir", "reports_dir"):
        if key in artifacts:
            Path(artifacts[key]).mkdir(parents=True, exist_ok=True)
