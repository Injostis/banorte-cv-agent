"""Construye una superficie A2UI (https://a2ui.org, v0.9.1) con los trofeos
platino de PlayStation.

El resultado se envuelve como un CallToolResult de MCP: un content part de
texto (fallback en prosa, autocontenido) y uno de tipo "resource" (la
convención EmbeddedResource de MCP) cuyo `resource.text` es la lista de
mensajes A2UI serializada a JSON string -- no una lista anidada.

Cada superficie son exactamente 3 mensajes: createSurface, updateComponents
(los componentes referencian datos con {"path": "..."}, no con valores
literales) y updateDataModel (los valores reales, publicados aparte). Solo
se usan componentes del catálogo básico v0.9.1 -- no existe un componente
de gráfica ahí, así que el ranking de rareza se muestra como tabla
(List + Text) en vez de una barra improvisada.
"""

import json
from typing import Any

A2UI_MIME_TYPE = "application/a2ui+json"
A2UI_VERSION = "v0.9.1"
BASIC_CATALOG = "https://a2ui.org/specification/v0_9_1/catalogs/basic/catalog.json"


def _create(surface_id: str, catalog_id: str) -> dict[str, Any]:
    return {"version": A2UI_VERSION, "createSurface": {"surfaceId": surface_id, "catalogId": catalog_id}}


def _components(surface_id: str, components: list[dict[str, Any]]) -> dict[str, Any]:
    return {"version": A2UI_VERSION, "updateComponents": {"surfaceId": surface_id, "components": components}}


def _data(surface_id: str, path: str, value: Any) -> dict[str, Any]:
    return {"version": A2UI_VERSION, "updateDataModel": {"surfaceId": surface_id, "path": path, "value": value}}


def _tool_result(fallback: str, uri: str, messages: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "content": [
            {"type": "text", "text": fallback},
            {
                "type": "resource",
                "resource": {"uri": uri, "mimeType": A2UI_MIME_TYPE, "text": json.dumps(messages, ensure_ascii=False)},
            },
        ]
    }


def _rarity_value(trofeo: dict[str, Any]) -> float:
    try:
        return float(trofeo.get("porcentaje_jugadores_con_este_trofeo") or 100)
    except (TypeError, ValueError):
        return 100.0


def build_ps_trophies_tool_result(trofeos: list[dict[str, Any]]) -> dict[str, Any]:
    """Arma el CallToolResult completo (fallback + superficie A2UI) para
    los 5 platinos más raros, como una tabla."""
    surface = "ps_trophies"
    top = sorted(trofeos, key=_rarity_value)[:5]

    components: list[dict[str, Any]] = [
        {"id": "root", "component": "Card", "child": "col"},
        {"id": "col", "component": "Column", "children": ["title", "list"]},
        {
            "id": "title",
            "component": "Text",
            "variant": "h3",
            "text": "Platinos más raros de Rodrigo en PlayStation",
        },
        {
            "id": "list",
            "component": "List",
            "direction": "vertical",
            "children": {"path": "/ps_trophies/items", "componentId": "row"},
        },
        {"id": "row", "component": "Row", "children": ["juego", "rareza"]},
        # Dentro del template de "list", los paths son relativos al item de la
        # iteración -- sin "/" inicial. Con "/" se leerían como ruta absoluta
        # desde la raíz del modelo de datos, donde no existen.
        {"id": "juego", "component": "Text", "variant": "body", "text": {"path": "juego"}, "weight": 3},
        {"id": "rareza", "component": "Text", "variant": "caption", "text": {"path": "rareza"}, "weight": 1},
    ]

    items = [
        {
            "juego": f"{trofeo.get('juego', '')} ({trofeo.get('plataforma', '')})",
            "rareza": f"{trofeo.get('porcentaje_jugadores_con_este_trofeo', '?')}%",
        }
        for trofeo in top
    ]

    messages = [
        _create(surface, BASIC_CATALOG),
        _components(surface, components),
        _data(surface, "/ps_trophies", {"items": items}),
    ]

    destacados = ", ".join(
        f"{t.get('juego', '')} ({t.get('porcentaje_jugadores_con_este_trofeo', '?')}%)" for t in top[:3]
    )
    fallback = (
        f"Mis 5 platinos más raros en PlayStation, por porcentaje de jugadores que también lo tienen: {destacados}, "
        "entre otros."
    )

    return _tool_result(fallback, f"a2ui://banorte-cv-agent/{surface}", messages)
