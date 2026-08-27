"""Construye mensajes de A2UI (https://a2ui.org, v0.9.1) con los trofeos
platino de PlayStation, para que la plataforma de Banorte los renderice en
su chat real.

Experimental: se manda como una parte adicional del contenido del mensaje,
junto al texto normal (que siempre está y siempre se ve, pase lo que pase
con esto -- ver app/responses_schema.py: build_response). El formato de
detección es el de la extensión A2A para A2UI (DataPart con
metadata.mimeType = "application/a2ui+json" -- ojo con el orden de las
palabras, es fácil confundirlo), declarada también en
/.well-known/agent-card.json.

Todos los mensajes del protocolo (createSurface, updateComponents, ...)
van juntos como una lista dentro del campo "data" de una sola Part -- por
especificación esa lista "no es una unidad transaccional", pero sí es una
sola Part, no una por mensaje.

No hay un componente de gráfica en el catálogo básico de A2UI (Text,
Image, Card, Column, Row, Button, etc. -- nada de "Chart"), así que la
"gráfica" es honesta: tarjetas con el dato de rareza como texto, no barras
dibujadas con precisión.
"""

from typing import Any

CATALOG_URL = "https://a2ui.org/specification/v0_9_1/catalogs/basic/catalog.json"
A2UI_MIME_TYPE = "application/a2ui+json"


def _rarity_value(trofeo: dict[str, Any]) -> float:
    try:
        return float(trofeo.get("porcentaje_jugadores_con_este_trofeo") or 100)
    except (TypeError, ValueError):
        return 100.0


def build_ps_trophies_messages(trofeos: list[dict[str, Any]], surface_id: str = "ps_trophies") -> list[dict[str, Any]]:
    """Arma los mensajes A2UI (createSurface + updateComponents) para un
    surface con los 5 platinos más raros, como tarjetas. Cada elemento de
    la lista es un mensaje independiente, tal como los pide el protocolo."""
    top = sorted(trofeos, key=_rarity_value)[:5]

    components: list[dict[str, Any]] = [
        {
            "id": "root",
            "component": "Column",
            "children": ["title", *[f"card_{i}" for i in range(len(top))]],
            "align": "stretch",
        },
        {
            "id": "title",
            "component": "Text",
            "text": "Platinos más raros de Rodrigo en PlayStation",
            "variant": "h3",
        },
    ]

    for i, trofeo in enumerate(top):
        col_id = f"col_{i}"
        components.append({"id": f"card_{i}", "component": "Card", "child": col_id})
        components.append(
            {"id": col_id, "component": "Column", "children": [f"game_{i}", f"detail_{i}", f"rarity_{i}"]}
        )
        components.append(
            {"id": f"game_{i}", "component": "Text", "text": str(trofeo.get("juego", "")), "variant": "h4"}
        )
        components.append(
            {
                "id": f"detail_{i}",
                "component": "Text",
                "text": f"{trofeo.get('plataforma', '')} · {trofeo.get('nombre_trofeo', '')}",
            }
        )
        components.append(
            {
                "id": f"rarity_{i}",
                "component": "Text",
                "text": f"Solo el {trofeo.get('porcentaje_jugadores_con_este_trofeo', '?')}% de los jugadores lo tiene",
            }
        )

    return [
        {"version": "v0.9.1", "createSurface": {"surfaceId": surface_id, "catalogId": CATALOG_URL}},
        {"version": "v0.9.1", "updateComponents": {"surfaceId": surface_id, "components": components}},
    ]


def wrap_as_a2a_data_part(messages: list[dict[str, Any]]) -> dict[str, Any]:
    """Envuelve la lista completa de mensajes A2UI en una sola Part de A2A.

    Por especificación, "data" es una lista de mensajes (no es una unidad
    transaccional), y el mimeType exacto que la identifica es
    "application/a2ui+json" -- ojo con el orden, es fácil confundirlo con
    "application/json+a2ui" (el error que tenía antes)."""
    return {
        "type": "data",
        "kind": "data",
        "metadata": {"mimeType": A2UI_MIME_TYPE},
        "data": messages,
    }
