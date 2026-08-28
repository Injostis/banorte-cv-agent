"""Descarga los trofeos platino desde PlayStation Network y los guarda como
snapshot en data/ps_trophies.json, que la tool get_ps_trophies lee.

Uso:
    1. Inicia sesión en https://my.playstation.com/
    2. En otra pestaña (misma sesión), visita
       https://ca.account.sony.com/api/v1/ssocookie y copia el valor de
       "npsso" (un código de 64 caracteres).
    3. Corre (el valor nunca se guarda en ningún lado por este script):

           PSN_NPSSO=<tu-npsso> uv run python scripts/fetch_ps_trophies.py

       o ponlo temporalmente como PSN_NPSSO=... en tu .env local (que ya
       está en .gitignore) y solo corre:

           uv run python scripts/fetch_ps_trophies.py
"""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from psnawp_api import PSNAWP
from psnawp_api.models.client import Client
from psnawp_api.models.trophies.trophy import TrophyWithProgress
from psnawp_api.models.trophies.trophy_constants import PlatformType, TrophyType

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "ps_trophies.json"


def _find_earned_platinum(
    client: Client, np_communication_id: str, platforms: frozenset[PlatformType]
) -> tuple[TrophyWithProgress, PlatformType] | None:
    """Recorre las plataformas dadas y regresa el trofeo platino ganado (y
    su plataforma) en la primera que lo tenga, o None si ninguna lo tiene."""
    last_error: Exception | None = None
    for platform in platforms:
        try:
            for trophy in client.trophies(np_communication_id, platform, include_progress=True):
                if trophy.trophy_type == TrophyType.PLATINUM and trophy.earned:
                    return trophy, platform
        except Exception as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise last_error
    return None


def main() -> None:
    load_dotenv()
    npsso = os.environ.get("PSN_NPSSO")
    if not npsso:
        print(
            "Falta PSN_NPSSO. Consigue tu npsso en "
            "https://ca.account.sony.com/api/v1/ssocookie (con sesión iniciada en "
            "my.playstation.com) y corre:\n"
            "  PSN_NPSSO=<valor> uv run python scripts/fetch_ps_trophies.py",
            file=sys.stderr,
        )
        raise SystemExit(1)

    client = PSNAWP(npsso).me()

    platinos = []
    for title in client.trophy_titles(limit=None):
        if title.earned_trophies.platinum < 1:
            continue

        if not title.title_platform or title.np_communication_id is None:
            continue

        try:
            result = _find_earned_platinum(client, title.np_communication_id, title.title_platform)
        except Exception as exc:
            print(f"SKIP: {title.title_name} ({exc.__class__.__name__})", file=sys.stderr)
            continue

        if result is None:
            continue
        platino, platform = result

        platinos.append(
            {
                "juego": title.title_name,
                "plataforma": platform.name,
                "progreso_total_del_juego": title.progress,
                "fecha_platino": (platino.earned_date_time.isoformat() if platino.earned_date_time else None),
                "nombre_trofeo": platino.trophy_name,
                "porcentaje_jugadores_con_este_trofeo": platino.trophy_earn_rate,
            }
        )
        print(f"OK: {title.title_name}")

    platinos.sort(key=lambda p: p["porcentaje_jugadores_con_este_trofeo"] or 100.0)

    OUTPUT_PATH.write_text(json.dumps(platinos, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{len(platinos)} platinos guardados en {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
