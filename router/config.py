from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"


@dataclass(frozen=True)
class ModelSpec:
    provider: str
    id: str
    usd_per_m_input: float
    usd_per_m_output: float


@dataclass(frozen=True)
class Config:
    jev_model: str
    small_model: ModelSpec
    large_model: ModelSpec
    max_output_tokens: int
    temperature: float
    seed: int
    n_prompts: int
    max_usd: float
    jev_usd_per_m_input: float = 0.042
    jev_phrasing: str = "statement"  # a key of router.jev.PHRASINGS, chosen on dev
    ollama_url: str = "http://localhost:11434"
    threshold: float = 0.5
    jev_timeout_s: float = 2.0
    # Cascade: the small model answers first, Jev checks the answer, the large model is used
    # only when a check fires. cascade_checks are keys of router.jev.CHECKS, chosen on dev.
    cascade_checks: tuple[str, ...] = ("wrong_answer",)
    cascade_threshold: float = 0.5


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for role in ("small_model", "large_model"):
        raw[role] = ModelSpec(**raw[role])
    if "cascade_checks" in raw:
        raw["cascade_checks"] = tuple(raw["cascade_checks"])
    return Config(**raw)
