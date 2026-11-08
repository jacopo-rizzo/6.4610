"""Loading YAML configs.

Configs live in `configs/` at the repo root and refer to each other by paths
relative to that folder (e.g. `base_tokenizer: tokenizer/llama3.yaml`).
`${oc.env:VAR}` values are filled from the environment, which is first
populated from the repo's `.env` file if one exists.
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


def load_env() -> None:
    """Read `.env` at the repo root into os.environ (existing vars win)."""
    load_dotenv(REPO_ROOT / ".env", override=False)


def resolve_config_path(path: str | Path) -> Path:
    """Accept either a path relative to `configs/` or any existing path."""
    path = Path(path)
    if path.exists():
        return path
    candidate = CONFIG_DIR / path
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Config not found: {path} (also looked in {CONFIG_DIR})")


def load_config(path: str | Path, overrides: list[str] | None = None) -> DictConfig:
    """Load a YAML config, apply its `defaults:` file (if any) and CLI overrides.

    `defaults: other.yaml` is resolved relative to the config's own folder, and
    the config's own values take precedence over the defaults.
    `overrides` are dotlist strings such as ["optim.lr=3e-4"].
    """
    load_env()
    path = resolve_config_path(path)
    cfg = OmegaConf.load(path)

    if "defaults" in cfg:
        base = load_config(path.parent / cfg.pop("defaults"))
        cfg = OmegaConf.merge(base, cfg)

    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return cfg
