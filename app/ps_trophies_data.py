"""Carga data/ps_trophies.json, el snapshot de trofeos platino de PlayStation
generado por scripts/fetch_ps_trophies.py."""

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
