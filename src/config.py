"""
Minimal configuration loader for RAG Document Q&A.
Loads settings from configs/default.yaml with environment variable overrides.
"""

import os
from pathlib import Path
from typing import Any

import yaml

_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "configs" / "default.yaml"

_config: dict | None = None


def load_config(path: str | None = None) -> dict:
    """Load configuration from YAML, cached after first call."""
    global _config
    if _config is not None:
        return _config

    config_path = Path(path) if path else _DEFAULT_CONFIG_PATH
    if config_path.exists():
        with open(config_path, "r") as f:
            _config = yaml.safe_load(f) or {}
    else:
        _config = {}

    return _config


_ENV_MAPPINGS = {
    "ingestion.embedding.provider": "EMBEDDING_PROVIDER",
    "ingestion.embedding.model": "EMBEDDING_MODEL",
    "generation.provider": "LLM_PROVIDER",
    "generation.model": "LLM_MODEL",
    "generation.temperature": "LLM_TEMPERATURE",
    "generation.max_tokens": "LLM_MAX_TOKENS",
    "retrieval.top_k": "RETRIEVAL_TOP_K",
    "retrieval.vector_store": "VECTOR_STORE",
    "api.host": "API_HOST",
    "api.port": "API_PORT",
    "app.max_file_size_mb": "MAX_FILE_SIZE_MB",
}


def _cast_env_value(env_value: str, yaml_value: Any) -> Any:
    """Cast an env var string to the type of the YAML config value."""
    if yaml_value is None:
        return env_value
    if isinstance(yaml_value, bool):
        return env_value.lower() in ("true", "1", "yes")
    if isinstance(yaml_value, int):
        try:
            return int(env_value)
        except ValueError:
            return env_value
    if isinstance(yaml_value, float):
        try:
            return float(env_value)
        except ValueError:
            return env_value
    return env_value


def get(key_path: str, default: Any = None) -> Any:
    """
    Get a config value by dot-separated path, with env var override.

    Environment variables take precedence. The env var name is looked up
    in an explicit mapping (e.g., 'generation.model' -> LLM_MODEL).
    If not mapped, the full key path is uppercased with dots replaced
    by underscores (e.g., 'retrieval.top_k' -> RETRIEVAL_TOP_K).

    The env var value is automatically cast to match the YAML value's type
    (int, float, bool, or str).

    Examples:
        get("ingestion.chunking.chunk_size")  -> 512
        get("retrieval.top_k")                -> 5
        # With RETRIEVAL_TOP_K=10 in env:
        get("retrieval.top_k")                -> 10
    """
    # Step 1: Resolve from YAML
    config = load_config()
    keys = key_path.split(".")
    yaml_value = config
    for key in keys:
        if isinstance(yaml_value, dict):
            yaml_value = yaml_value.get(key)
        else:
            yaml_value = None
            break
        if yaml_value is None:
            break

    # Step 2: Check environment variable override
    env_key = _ENV_MAPPINGS.get(key_path)
    if env_key is None:
        # Fallback: uppercase the full key path with underscores
        env_key = key_path.replace(".", "_").upper()

    env_value = os.getenv(env_key)
    if env_value is not None:
        return _cast_env_value(
            env_value, yaml_value if yaml_value is not None else default
        )

    # Step 3: Return YAML value or default
    return yaml_value if yaml_value is not None else default
