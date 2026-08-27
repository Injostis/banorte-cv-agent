"""Carga profile.yaml como la única fuente de verdad del agente.

Las tools en app/tools.py leen únicamente de aquí -- nunca del conocimiento
general del modelo. Si un dato no está en profile.yaml, no existe para el
agente.
"""

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
