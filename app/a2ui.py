"""Construye superficies A2UI (https://a2ui.org, v0.9.1) para las tools que
regresan resultados visuales, en vez de solo texto.

El resultado de cada superficie se envuelve como un CallToolResult de MCP:
un content part de texto (fallback en prosa) y uno de tipo "resource" cuyo
`resource.text` es la lista de mensajes A2UI serializada a JSON string.

Cada superficie son exactamente 3 mensajes: createSurface, updateComponents
y updateDataModel.

Los componentes usados vienen del catálogo básico v0.9.1: un ranking se
muestra como tabla (dos columnas de texto) y un nivel de dominio se
representa con iconos de estrella ("star"/"starOff").
"""

import json
import uuid
from datetime import datetime
from typing import Any

A2UI_MIME_TYPE = "application/a2ui+json"
A2UI_VERSION = "v0.9.1"
BASIC_CATALOG = "https://a2ui.org/specification/v0_9_1/catalogs/basic/catalog.json"

_NIVEL_SCORE = {"avanzado": 3, "intermedio": 2, "basico": 1}
_NIVEL_LABEL = {"avanzado": "Avanzado", "intermedio": "Intermedio", "basico": "Básico"}
_MESES_ES = (
    "ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic",
)


def _new_surface_id(prefix: str) -> str:
    """Genera un id de superficie único, combinando el prefijo dado con un
    sufijo aleatorio."""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


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


def _fecha_corta(fecha_iso: str) -> str:
    try:
        dt = datetime.fromisoformat(fecha_iso)
    except (TypeError, ValueError):
        return ""
    return f"{_MESES_ES[dt.month - 1]} {dt.year}"


def build_ps_trophies_tool_result(trofeos: list[dict[str, Any]]) -> dict[str, Any]:
    """Arma el CallToolResult completo (fallback + superficie A2UI) para
    los trofeos platino: total, y una tabla de los 5 más raros con el
    nombre del trofeo y la fecha en que lo obtuvo."""
    surface = _new_surface_id("ps_trophies")
    top = sorted(trofeos, key=_rarity_value)[:5]

    names_children = [f"juego{i}" if j == 0 else f"meta{i}" for i in range(len(top)) for j in range(2)]
    pcts_children = [f"rareza{i}" if j == 0 else f"blank{i}" for i in range(len(top)) for j in range(2)]

    components: list[dict[str, Any]] = [
        {"id": "root", "component": "Card", "child": "col"},
        {"id": "col", "component": "Column", "children": ["title", "total", "subtitle", "header", "sep", "table"]},
        {"id": "title", "component": "Text", "variant": "h3", "text": "Trofeos platino de Rodrigo en PlayStation"},
        {"id": "total", "component": "Text", "variant": "body", "text": f"{len(trofeos)} platinos en total."},
        {
            "id": "subtitle",
            "component": "Text",
            "variant": "caption",
            "text": (
                "Los 5 más raros -- % = jugadores en el mundo que también tienen ese trofeo, entre más bajo, "
                "más raro."
            ),
        },
        {"id": "header", "component": "Row", "justify": "spaceBetween", "children": ["header_juego", "header_pct"]},
        {"id": "header_juego", "component": "Text", "variant": "caption", "text": "Juego", "weight": 3},
        {"id": "header_pct", "component": "Text", "variant": "caption", "text": "% con el trofeo", "weight": 1},
        {"id": "sep", "component": "Divider"},
        {"id": "table", "component": "Row", "justify": "spaceBetween", "children": ["names_col", "pcts_col"]},
        {"id": "names_col", "component": "Column", "children": names_children, "weight": 3},
        {"id": "pcts_col", "component": "Column", "children": pcts_children, "weight": 1},
    ]

    for i, trofeo in enumerate(top):
        components.extend(
            [
                {
                    "id": f"juego{i}",
                    "component": "Text",
                    "variant": "body",
                    "text": f"{trofeo.get('juego', '')} ({trofeo.get('plataforma', '')})",
                },
                {
                    "id": f"meta{i}",
                    "component": "Text",
                    "variant": "caption",
                    "text": (
                        f"\"{trofeo.get('nombre_trofeo', '')}\" · {_fecha_corta(trofeo.get('fecha_platino', ''))}"
                    ),
                },
                {
                    "id": f"rareza{i}",
                    "component": "Text",
                    "variant": "caption",
                    "text": f"{trofeo.get('porcentaje_jugadores_con_este_trofeo', '?')}%",
                },
                {"id": f"blank{i}", "component": "Text", "variant": "caption", "text": " "},
            ]
        )

    messages = [
        _create(surface, BASIC_CATALOG),
        _components(surface, components),
        _data(surface, "/", {}),
    ]

    destacados = ", ".join(
        f"{t.get('juego', '')} ({t.get('porcentaje_jugadores_con_este_trofeo', '?')}%)" for t in top[:3]
    )
    fallback = (
        f"Tengo {len(trofeos)} platinos en total. Los 5 más raros, por porcentaje de jugadores que también lo "
        f"tienen: {destacados}, entre otros."
    )

    return _tool_result(fallback, f"a2ui://banorte-cv-agent/{surface}", messages)


def build_profile_card_tool_result(profile: dict[str, Any]) -> dict[str, Any]:
    """Arma el CallToolResult con la tarjeta de perfil: nombre, título,
    resumen, skills avanzadas, correo como texto, y botones de GitHub/
    LinkedIn."""
    surface = _new_surface_id("profile")
    contacto = profile["contacto"]
    avanzadas = [h["nombre"] for h in profile.get("habilidades_destacadas", []) if h.get("nivel") == "avanzado"]
    skills_text = ", ".join(avanzadas)

    components: list[dict[str, Any]] = [
        {"id": "root", "component": "Card", "child": "col"},
        {
            "id": "col",
            "component": "Column",
            "children": ["name", "headline", "summary", "sep", "skills_title", "skills", "email", "sep2", "links"],
        },
        {"id": "name", "component": "Text", "variant": "h3", "text": {"path": "/profile/name"}},
        {"id": "headline", "component": "Text", "variant": "caption", "text": {"path": "/profile/headline"}},
        {"id": "summary", "component": "Text", "variant": "body", "text": {"path": "/profile/summary"}},
        {"id": "sep", "component": "Divider"},
        {"id": "skills_title", "component": "Text", "variant": "h5", "text": "Skills avanzadas"},
        {"id": "skills", "component": "Text", "variant": "body", "text": {"path": "/profile/skills"}},
        {"id": "email", "component": "Text", "variant": "caption", "text": {"path": "/profile/email_texto"}},
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
                "email_texto": f"Correo: {contacto['email']}",
            },
        ),
    ]

    fallback = (
        f"{profile['nombre']} -- {headline}. {profile['resumen'].strip()} "
        f"Skills avanzadas: {skills_text}. Correo: {contacto['email']}. GitHub: {contacto['repo_github']}. "
        f"LinkedIn: {contacto['linkedin']}."
    )
    return _tool_result(fallback, f"a2ui://banorte-cv-agent/{surface}", messages)


def build_contact_card_tool_result(profile: dict[str, Any]) -> dict[str, Any]:
    """Arma el CallToolResult con una tarjeta chica de contacto: nombre,
    correo como texto, y botones de GitHub/LinkedIn."""
    surface = _new_surface_id("contact")
    contacto = profile["contacto"]

    components: list[dict[str, Any]] = [
        {"id": "root", "component": "Card", "child": "col"},
        {"id": "col", "component": "Column", "children": ["name", "email", "sep", "links"]},
        {"id": "name", "component": "Text", "variant": "h5", "text": {"path": "/contact/name"}},
        {"id": "email", "component": "Text", "variant": "body", "text": {"path": "/contact/email_texto"}},
        {"id": "sep", "component": "Divider"},
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

    messages = [
        _create(surface, BASIC_CATALOG),
        _components(surface, components),
        _data(
            surface,
            "/contact",
            {"name": profile["nombre"], "email_texto": f"Correo: {contacto['email']}"},
        ),
    ]

    fallback = f"Correo: {contacto['email']}. GitHub: {contacto['repo_github']}. LinkedIn: {contacto['linkedin']}."
    return _tool_result(fallback, f"a2ui://banorte-cv-agent/{surface}", messages)


def _star_icons(score: int) -> dict[str, str]:
    return {f"icon{i}": ("star" if i <= score else "starOff") for i in (1, 2, 3)}


def build_skills_levels_tool_result(habilidades_destacadas: list[dict[str, Any]]) -> dict[str, Any]:
    """Arma el CallToolResult con el panorama visual de skills destacadas
    por nivel de dominio, usando 3 iconos de estrella por skill (llenas
    según el nivel)."""
    surface = _new_surface_id("skills")

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
        # Dentro del template, los paths son relativos al item -- sin "/" inicial,
        # incluyendo los de "stars" (subárbol anidado dentro del template).
        {"id": "skill_row", "component": "Row", "justify": "spaceBetween", "children": ["skill_name", "stars"]},
        {"id": "skill_name", "component": "Text", "variant": "body", "text": {"path": "nombre_nivel"}, "weight": 3},
        {"id": "stars", "component": "Row", "justify": "end", "children": ["star1", "star2", "star3"], "weight": 1},
        {"id": "star1", "component": "Icon", "name": {"path": "icon1"}},
        {"id": "star2", "component": "Icon", "name": {"path": "icon2"}},
        {"id": "star3", "component": "Icon", "name": {"path": "icon3"}},
    ]

    ordered = sorted(habilidades_destacadas, key=lambda h: -_NIVEL_SCORE.get(h.get("nivel", ""), 0))
    items: list[dict[str, Any]] = []
    for h in ordered:
        score = _NIVEL_SCORE.get(h.get("nivel", ""), 0)
        nivel_texto = _NIVEL_LABEL.get(h.get("nivel", ""), h.get("nivel", ""))
        items.append({"nombre_nivel": f"{h['nombre']} — {nivel_texto}", **_star_icons(score)})

    messages = [
        _create(surface, BASIC_CATALOG),
        _components(surface, components),
        _data(surface, "/skills", {"items": items}),
    ]

    detalle = ", ".join(item["nombre_nivel"] for item in items)
    fallback = f"Un vistazo de mis skills destacadas por nivel de dominio: {detalle}."

    return _tool_result(fallback, f"a2ui://banorte-cv-agent/{surface}", messages)
