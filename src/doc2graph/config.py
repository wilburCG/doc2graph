"""Configuration management."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel


class Neo4jConfig(BaseModel):
    uri: str = "bolt://localhost:7687"
    username: str = "neo4j"
    password: str = ""
    database: str = "neo4j"


class LlmConfig(BaseModel):
    provider: str = "openai"
    model: str = "gpt-4o"
    api_key: str = ""
    api_base: str = "https://api.openai.com/v1"


class ExtractionConfig(BaseModel):
    batch_size: int = 5
    max_entities: int = 50
    max_relations: int = 100
    temperature: float = 0.1


class MergeConfig(BaseModel):
    strategy: str = "name_type"
    threshold: float = 0.85


class Config(BaseModel):
    """Application configuration."""
    neo4j: Neo4jConfig = Neo4jConfig()
    llm: LlmConfig = LlmConfig()
    extraction: ExtractionConfig = ExtractionConfig()
    merge: MergeConfig = MergeConfig()

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        """Load configuration from a YAML file."""
        if path is None:
            path = Path("config.yaml")
        p = Path(path)
        if not p.exists():
            return cls()

        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        # Resolve environment variable references like ${VAR}
        data = _resolve_env(data)

        return cls.model_validate(data)

    def save(self, path: str | Path = "config.yaml") -> None:
        """Save configuration to a YAML file."""
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, allow_unicode=True)


DEFAULT_CONFIG_YAML = """neo4j:
  uri: bolt://localhost:7687
  username: neo4j
  password: your_password

llm:
  provider: openai
  model: gpt-4o
  api_key: ${OPENAI_API_KEY}
  api_base: https://api.openai.com/v1

extraction:
  batch_size: 5
  max_entities: 50
  max_relations: 100
  temperature: 0.1

merge:
  strategy: name_type
  threshold: 0.85
"""


def _resolve_env(data: Any) -> Any:
    """Recursively resolve ${VAR} patterns in config values."""
    if isinstance(data, dict):
        return {k: _resolve_env(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_resolve_env(item) for item in data]
    if isinstance(data, str) and data.startswith("${") and data.endswith("}"):
        var_name = data[2:-1]
        return os.environ.get(var_name, "")
    return data
