"""Construye un surface de A2UI (https://a2ui.org, v0.9.1) con los trofeos
platino de PlayStation, para probar si la plataforma de Banorte lo
renderiza en su chat real.

Experimental: se manda como una parte adicional del contenido del mensaje,
junto al texto normal (que siempre está y siempre se ve, pase lo que pase
con esto -- ver app/responses_schema.py: build_response). El formato de
detección es el de la extensión A2A para A2UI (DataPart con
metadata.mimeType = "application/json+a2ui"), declarada también en
/.well-known/agent-card.json -- sin esa declaración de capacidad, un
cliente que sí sabe renderizar A2UI podría no buscarlo siquiera. Si la
plataforma no sabe interpretarlo, no debería romper nada más que esta
parte; y si sí se ve mal, este archivo es lo único que hay que quitar
para revertir el experimento.

No hay un componente de gráfica en el catálogo básico de A2UI (Text,
Image, Card, Column, Row, Button, etc. -- nada de "Chart"), así que la
"gráfica" es honesta: tarjetas con el dato de rareza como texto, no barras
dibujadas con precisión.
"""

from typing import Any

CATALOG_URL = "https://a2ui.org/specification/v0_9_1/catalogs/basic/catalog.json"


def _rarity_value(trofeo: dict[str, Any]) -> float:
    try:
        return float(trofeo.get("porcentaje_jugadores_con_este_trofeo") or 100)
    except (TypeError, ValueError):
        return 100.0


def build_ps_trophies_surface(trofeos: list[dict[str, Any]], surface_id: str = "ps_trophies") -> dict[str, Any]:
    """Arma un surface A2UI con los 5 platinos más raros, como tarjetas."""
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

    # Formato de DataPart de la extensión A2A para A2UI: el mecanismo de
    # detección real es el mimeType en metadata, no un campo "type" propio
    # -- se incluye "type": "data" también por si el cliente espera un
    # content part discriminado al estilo Open Responses; no hace daño
    # tenerlo de más.
    return {
        "type": "data",
        "kind": "data",
        "metadata": {"mimeType": "application/json+a2ui"},
        "data": {
            "version": "v0.9.1",
            "messages": [
                {"version": "v0.9.1", "createSurface": {"surfaceId": surface_id, "catalogId": CATALOG_URL}},
                {"version": "v0.9.1", "updateComponents": {"surfaceId": surface_id, "components": components}},
            ],
        },
    }
