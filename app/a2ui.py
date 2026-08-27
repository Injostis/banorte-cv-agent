"""Construye superficies A2UI (https://a2ui.org, v0.9.1) para las tools que
regresan resultados visuales, en vez de solo texto.

El resultado de cada superficie se envuelve como un CallToolResult de MCP: un
content part de texto (fallback en prosa, autocontenido) y uno de tipo
"resource" (la convención EmbeddedResource de MCP) cuyo `resource.text` es la
lista de mensajes A2UI serializada a JSON string -- no una lista anidada, y el
CallToolResult completo se serializa a JSON string una vez más al armar la
respuesta (ver app/responses_schema.py) -- así sobrevive intacto sin importar
qué capa lo transporte.

Cada superficie son exactamente 3 mensajes: createSurface, updateComponents
(los componentes referencian datos con {"path": "..."}, no con valores
literales) y updateDataModel (los valores reales, publicados aparte). Dentro
de un template de lista (children con "componentId"+"path"), los paths de los
componentes hijos son relativos al item de la iteración -- sin "/" inicial.

Solo se usan componentes del catálogo básico v0.9.1 -- no existe un
componente de gráfica ahí, así que un ranking se muestra como tabla
(List + Text) y un nivel de dominio se aproxima con un Slider de solo
lectura (el componente más parecido a una barra que ofrece el catálogo).
"""

import json
from typing import Any

A2UI_MIME_TYPE = "application/a2ui+json"
A2UI_VERSION = "v0.9.1"
BASIC_CATALOG = "https://a2ui.org/specification/v0_9_1/catalogs/basic/catalog.json"

_NIVEL_SCORE = {"avanzado": 3, "intermedio": 2, "basico": 1}
_NIVEL_LABEL = {"avanzado": "Avanzado", "intermedio": "Intermedio", "basico": "Básico"}


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
    los 5 platinos más raros, como una tabla con encabezado."""
    surface = "ps_trophies"
    top = sorted(trofeos, key=_rarity_value)[:5]

    components: list[dict[str, Any]] = [
        {"id": "root", "component": "Card", "child": "col"},
        {"id": "col", "component": "Column", "children": ["title", "subtitle", "header", "sep", "list"]},
        {"id": "title", "component": "Text", "variant": "h3", "text": "Platinos más raros de Rodrigo en PlayStation"},
        {
            "id": "subtitle",
            "component": "Text",
            "variant": "caption",
            "text": "% = jugadores en el mundo que también tienen ese trofeo -- entre más bajo, más raro.",
        },
        {"id": "header", "component": "Row", "justify": "spaceBetween", "children": ["header_juego", "header_pct"]},
        {"id": "header_juego", "component": "Text", "variant": "caption", "text": "Juego", "weight": 3},
        {"id": "header_pct", "component": "Text", "variant": "caption", "text": "% con el trofeo", "weight": 1},
        {"id": "sep", "component": "Divider"},
        {
            "id": "list",
            "component": "List",
            "direction": "vertical",
            "children": {"path": "/ps_trophies/items", "componentId": "row"},
        },
        {"id": "row", "component": "Row", "justify": "spaceBetween", "children": ["juego", "rareza"]},
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


def build_profile_card_tool_result(profile: dict[str, Any]) -> dict[str, Any]:
    """Arma el CallToolResult con la tarjeta de perfil: nombre, título,
    resumen, skills avanzadas y botones de GitHub/LinkedIn."""
    surface = "profile"
    contacto = profile["contacto"]
    avanzadas = [h["nombre"] for h in profile.get("habilidades_destacadas", []) if h.get("nivel") == "avanzado"]
    skills_text = ", ".join(avanzadas)

    components: list[dict[str, Any]] = [
        {"id": "root", "component": "Card", "child": "col"},
        {
            "id": "col",
            "component": "Column",
            "children": ["name", "headline", "summary", "sep", "skills_title", "skills", "sep2", "links"],
        },
        {"id": "name", "component": "Text", "variant": "h3", "text": {"path": "/profile/name"}},
        {"id": "headline", "component": "Text", "variant": "caption", "text": {"path": "/profile/headline"}},
        {"id": "summary", "component": "Text", "variant": "body", "text": {"path": "/profile/summary"}},
        {"id": "sep", "component": "Divider"},
        {"id": "skills_title", "component": "Text", "variant": "h5", "text": "Skills avanzadas"},
        {"id": "skills", "component": "Text", "variant": "body", "text": {"path": "/profile/skills"}},
        {"id": "sep2", "component": "Divider"},
        {"id": "links", "component": "Row", "justify": "start", "children": ["gh", "li"]},
        {
            "id": "gh",
            "component": "Button",
            "variant": "default",
            "child": "gh_t",
            "action": {"functionCall": {"call": "openUrl", "args": {"url": contacto["repo_github"]}}},
        },
        {"id": "gh_t", "component": "Text", "text": "GitHub"},
        {
            "id": "li",
            "component": "Button",
            "variant": "default",
            "child": "li_t",
            "action": {"functionCall": {"call": "openUrl", "args": {"url": contacto["linkedin"]}}},
        },
        {"id": "li_t", "component": "Text", "text": "LinkedIn"},
    ]

    headline = f"{profile['titulo']} · {contacto['ubicacion']}"
    messages = [
        _create(surface, BASIC_CATALOG),
        _components(surface, components),
        _data(
            surface,
            "/profile",
            {
                "name": profile["nombre"],
                "headline": headline,
                "summary": profile["resumen"].strip(),
                "skills": skills_text,
            },
        ),
    ]

    fallback = (
        f"{profile['nombre']} -- {headline}. {profile['resumen'].strip()} "
        f"Skills avanzadas: {skills_text}. GitHub: {contacto['repo_github']}. LinkedIn: {contacto['linkedin']}."
    )
    return _tool_result(fallback, f"a2ui://banorte-cv-agent/{surface}", messages)


def build_skills_levels_tool_result(habilidades_destacadas: list[dict[str, Any]]) -> dict[str, Any]:
    """Arma el CallToolResult con el panorama visual de skills destacadas
    por nivel de dominio, usando un Slider de solo lectura como pseudo-barra
    (el catálogo básico no trae un componente de gráfica)."""
    surface = "skills"

    components: list[dict[str, Any]] = [
        {"id": "root", "component": "Card", "child": "col"},
        {"id": "col", "component": "Column", "children": ["title", "subtitle", "list"]},
        {"id": "title", "component": "Text", "variant": "h3", "text": "Skills destacadas de Rodrigo por nivel"},
        {
            "id": "subtitle",
            "component": "Text",
            "variant": "caption",
            "text": "Nivel de dominio autoevaluado, no un score objetivo. Lista completa con `get_skills`.",
        },
        {
            "id": "list",
            "component": "List",
            "direction": "vertical",
            "children": {"path": "/skills/items", "componentId": "skill_row"},
        },
        {
            "id": "skill_row",
            "component": "Row",
            "justify": "spaceBetween",
            "children": ["skill_name", "skill_bar", "skill_level"],
        },
        {"id": "skill_name", "component": "Text", "variant": "body", "text": {"path": "nombre"}, "weight": 2},
        {"id": "skill_bar", "component": "Slider", "value": {"path": "score"}, "min": 0, "max": 3, "weight": 3},
        {"id": "skill_level", "component": "Text", "variant": "caption", "text": {"path": "nivel_texto"}, "weight": 1},
    ]

    ordered = sorted(habilidades_destacadas, key=lambda h: -_NIVEL_SCORE.get(h.get("nivel", ""), 0))
    items = [
        {
            "nombre": h["nombre"],
            "score": _NIVEL_SCORE.get(h.get("nivel", ""), 0),
            "nivel_texto": _NIVEL_LABEL.get(h.get("nivel", ""), h.get("nivel", "")),
        }
        for h in ordered
    ]

    messages = [
        _create(surface, BASIC_CATALOG),
        _components(surface, components),
        _data(surface, "/skills", {"items": items}),
    ]

    detalle = ", ".join(f"{item['nombre']} ({item['nivel_texto']})" for item in items)
    fallback = f"Un vistazo de mis skills destacadas por nivel de dominio: {detalle}."

    return _tool_result(fallback, f"a2ui://banorte-cv-agent/{surface}", messages)
