"""Carga data/ps_trophies.json -- el dato curioso/personal del agente.

A diferencia de profile.yaml, este archivo no se escribe a mano: lo genera
scripts/fetch_ps_trophies.py una sola vez, corrido localmente contra la API
de PlayStation Network. El servicio desplegado solo lee este snapshot, nunca
llama a PSN directamente.
"""

import json
from functools import lru_cache
from typing import Any

from app.config import settings


@lru_cache(maxsize=1)
def load_ps_trophies() -> list[dict[str, Any]]:
    with settings.ps_trophies_path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{settings.ps_trophies_path} no contiene una lista JSON válida en la raíz.")
    return data
