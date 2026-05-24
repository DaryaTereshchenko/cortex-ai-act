"""Centralized model registry — singleton cache for SentenceTransformer models.

All modules that need embedding models should use this registry instead of
loading models independently.  This avoids duplicate copies of the same model
in memory and provides explicit ``cleanup`` / ``unload`` helpers so the eval
pipeline can free GPU/CPU RAM between stages.
"""

from __future__ import annotations

import gc
import threading
from typing import Any

_lock = threading.Lock()
_models: dict[str, Any] = {}


def get_model(model_name: str, **kwargs: Any) -> Any:
    """Return a cached ``SentenceTransformer`` instance, loading it on first call."""
    with _lock:
        if model_name in _models:
            return _models[model_name]

    # Import lazily so modules that never call this don't need the dep.
    try:
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing dependency 'sentence-transformers'. "
            "Install with: pip install sentence-transformers"
        ) from exc

    model = SentenceTransformer(model_name, **kwargs)

    with _lock:
        # Double-check after acquiring the lock (another thread may have loaded it).
        if model_name not in _models:
            _models[model_name] = model
        return _models[model_name]


def unload(model_name: str) -> None:
    """Remove a single model from the registry and free its memory."""
    with _lock:
        model = _models.pop(model_name, None)
    if model is not None:
        del model
    _collect()


def cleanup() -> None:
    """Unload **all** models and aggressively free memory."""
    with _lock:
        _models.clear()
    _collect()


def loaded_models() -> list[str]:
    """Return the list of currently loaded model names."""
    with _lock:
        return list(_models)


def _collect() -> None:
    """Run garbage collection and clear CUDA caches if available."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
