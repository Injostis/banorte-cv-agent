"""Carga profile.yaml, el perfil de Rodrigo usado por las tools del agente."""

from functools import lru_cache
from typing import Any

import yaml

from app.config import settings


@lru_cache(maxsize=1)
def load_profile() -> dict[str, Any]:
    with settings.profile_path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{settings.profile_path} no contiene un mapeo YAML válido en la raíz.")
    return data
