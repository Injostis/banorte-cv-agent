"""Ingesta única de trofeos platino desde PlayStation Network.

Este script NO forma parte del servicio desplegado -- se corre a mano, en tu
máquina, una sola vez (o cada que quieras actualizar la lista), y guarda un
snapshot en data/ps_trophies.json que la tool get_ps_trophies lee en
producción. El agente desplegado nunca llama a la API de PSN: así se evita
tanto el riesgo de que un uso repetido/automatizado termine en una
suspensión de tu cuenta (la propia librería lo advierte), como depender de
un token que expira cada ~2 meses en el servicio que Banorte va a probar.

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
    """Prueba cada plataforma del título hasta encontrar el platino.

    Un título puede existir en más de una plataforma (ej. PS4 y PS5), y solo
    una de ellas suele tener el set de trofeos consultable -- si la primera
    que se prueba falla o no tiene el platino, se sigue con las demás antes
    de rendirse.
    """
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
            # Algunos títulos (ediciones retiradas de la tienda, colecciones
            # con metadata rota, etc.) truenan al pedir su detalle de
            # trofeos en TODAS sus plataformas -- un juego problemático no
            # debe perder el trabajo ya hecho en todos los demás.
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
