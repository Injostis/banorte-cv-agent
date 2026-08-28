"""Tools que el agente puede invocar sobre profile.yaml.

Cada tool es un lookup de solo lectura sobre una sección de profile.yaml.
"""

import logging
from typing import Any

from anthropic.types import ToolParam
from langfuse import observe

from app.a2ui import (
    build_contact_card_tool_result,
    build_profile_card_tool_result,
    build_project_architecture_tool_result,
    build_ps_trophies_tool_result,
    build_skills_levels_tool_result,
)
from app.profile_data import load_profile
from app.ps_trophies_data import load_ps_trophies

logger = logging.getLogger(__name__)

TOOL_SCHEMAS: list[ToolParam] = [
    {
        "name": "get_summary",
        "description": (
            "Obtiene el nombre, título profesional y resumen general de Rodrigo. "
            "Úsala para preguntas amplias como '¿quién eres?' o '¿a qué te dedicas?'."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_experience",
        "description": (
            "Obtiene el historial de experiencia laboral de Rodrigo "
            "(empresas, puestos, periodos y responsabilidades)."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_projects",
        "description": (
            "Obtiene los proyectos técnicos de Rodrigo (personales, de hackathon o de portafolio), "
            "con su descripción, stack y estado."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_skills",
        "description": (
            "Obtiene la lista completa y plana de habilidades técnicas (lenguajes, frameworks, herramientas) e "
            "idiomas de Rodrigo, sin nivel de dominio. Úsala SOLO si piden explícitamente la lista completa (ej. "
            "'dame todas tus skills'). Para cualquier otra pregunta de skills ('¿qué tecnologías usas?', '¿en qué "
            "tienes más nivel?', '¿qué tan bien manejas X?') usa mejor `show_skills_levels`."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_education",
        "description": "Obtiene la formación académica y los certificados de Rodrigo.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_contact",
        "description": (
            "Obtiene la información de contacto pública de Rodrigo (correo, ubicación, proyectos públicos con "
            "URL). Úsala SOLO para algo puntual que ni `show_contact_card` ni `show_profile_card` cubran, como "
            "preguntar por un proyecto público específico (ej. Muralea)."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_ps_trophies",
        "description": (
            "Obtiene la lista completa y detallada de trofeos platino de PlayStation de Rodrigo, uno por uno. Es "
            "un dato personal/curioso, no profesional -- úsala SOLO para una pregunta muy puntual sobre un juego "
            "específico que no esté entre los más raros (ej. '¿tienes el platino de X juego en particular?'). "
            "Para cualquier pregunta general sobre sus platinos ('¿has platinado juegos?', '¿cuántos platinos "
            "tienes?') o para ver cuáles son los más raros, usa mejor `show_ps_trophies_table`."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "show_ps_trophies_table",
        "description": (
            "Muestra una tabla visual con los 5 platinos más raros de Rodrigo en PlayStation (juego y porcentaje "
            "de jugadores que también lo tiene). Úsala para CUALQUIER pregunta sobre sus platinos de PlayStation, "
            "general o puntual (ej. '¿has platinado juegos?', '¿cuáles son tus más raros?') -- ya incluye tanto el "
            "total como el detalle visual, así que cubre la pregunta general sin necesitar un segundo turno. "
            "Muéstrala una sola vez por conversación -- si ya la mostraste, no la repitas, solo refiérete a ella."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "show_profile_card",
        "description": (
            "Muestra una tarjeta visual con el nombre, título, resumen, skills avanzadas y datos de contacto de "
            "Rodrigo. Úsala al inicio de la conversación (en tu primera respuesta, sin importar qué tan casual "
            "sea el mensaje de apertura) o cuando pidan una vista general de su perfil. Para preguntas puntuales "
            "de cómo contactarlo usa mejor `show_contact_card` -- es más chica, sin repetir el resumen completo. "
            "Muéstrala una sola vez por conversación -- si ya la mostraste, no la repitas, solo refiérete a ella."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "show_contact_card",
        "description": (
            "Muestra una tarjeta chica solo con el correo y botones de GitHub/LinkedIn de Rodrigo, sin resumen ni "
            "skills. Úsala para CUALQUIER pregunta puntual de cómo contactarlo ('¿cómo te contacto?', '¿cuál es "
            "tu correo?', '¿me compartes tu LinkedIn/GitHub?'). Si ya mostraste `show_profile_card` en esta "
            "conversación (que también incluye el correo y los botones), no hace falta usar esta -- solo "
            "refiérete a la que ya se mostró. Muéstrala una sola vez por conversación."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "show_skills_levels",
        "description": (
            "Muestra un panorama visual de las skills destacadas de Rodrigo agrupadas por nivel de dominio "
            "(avanzado/intermedio/básico), con estrellas por skill. Úsala para CUALQUIER pregunta "
            "sobre sus skills o stack técnico, general o puntual (ej. '¿qué tecnologías usas?', '¿en qué tienes "
            "más nivel?') -- solo usa `get_skills` si piden explícitamente la lista completa y plana, sin nivel. "
            "Muéstrala una sola vez por conversación -- si ya la mostraste, no la repitas."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "show_project_architecture",
        "description": (
            "Muestra una tarjeta visual con las decisiones técnicas detrás de este mismo agente "
            "(guardrails, A2UI, observabilidad, tools, diseño sin estado). Úsala cuando pregunten cómo está "
            "construido el agente, qué tecnologías usa, o por sus decisiones de arquitectura -- distinto de "
            "preguntas sobre la trayectoria profesional de Rodrigo. Muéstrala una sola vez por conversación."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
]


def _get_summary(_: dict[str, Any]) -> dict[str, Any]:
    profile = load_profile()
    return {
        "nombre": profile["nombre"],
        "titulo": profile["titulo"],
        "resumen": profile["resumen"],
    }


def _get_experience(_: dict[str, Any]) -> dict[str, Any]:
    return {"experiencia": load_profile()["experiencia"]}


def _get_projects(_: dict[str, Any]) -> dict[str, Any]:
    return {"proyectos": load_profile()["proyectos"]}


def _get_skills(_: dict[str, Any]) -> dict[str, Any]:
    return {"habilidades": load_profile()["habilidades"]}


def _get_education(_: dict[str, Any]) -> dict[str, Any]:
    profile = load_profile()
    return {"educacion": profile["educacion"], "certificados": profile["certificados"]}


def _get_contact(_: dict[str, Any]) -> dict[str, Any]:
    return {"contacto": load_profile()["contacto"]}


def _get_ps_trophies(_: dict[str, Any]) -> dict[str, Any]:
    trofeos = load_ps_trophies()
    return {"total_platinos": len(trofeos), "trofeos": trofeos}


def _show_ps_trophies_table(_: dict[str, Any]) -> dict[str, Any]:
    """Regresa un CallToolResult con la tabla de trofeos platino como
    superficie A2UI. Ver app/a2ui.py."""
    trofeos = load_ps_trophies()
    return build_ps_trophies_tool_result(trofeos)


def _show_profile_card(_: dict[str, Any]) -> dict[str, Any]:
    return build_profile_card_tool_result(load_profile())


def _show_contact_card(_: dict[str, Any]) -> dict[str, Any]:
    return build_contact_card_tool_result(load_profile())


def _show_skills_levels(_: dict[str, Any]) -> dict[str, Any]:
    habilidades_destacadas = load_profile().get("habilidades_destacadas", [])
    return build_skills_levels_tool_result(habilidades_destacadas)


def _show_project_architecture(_: dict[str, Any]) -> dict[str, Any]:
    return build_project_architecture_tool_result()


_DISPATCH = {
    "get_summary": _get_summary,
    "get_experience": _get_experience,
    "get_projects": _get_projects,
    "get_skills": _get_skills,
    "get_education": _get_education,
    "get_contact": _get_contact,
    "get_ps_trophies": _get_ps_trophies,
    "show_ps_trophies_table": _show_ps_trophies_table,
    "show_profile_card": _show_profile_card,
    "show_contact_card": _show_contact_card,
    "show_skills_levels": _show_skills_levels,
    "show_project_architecture": _show_project_architecture,
}


@observe(as_type="tool", name="execute_tool")
def execute_tool(name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
    """Ejecuta la tool indicada por nombre y regresa su resultado.

    Si el nombre no está registrado o la tool lanza una excepción, regresa
    un dict con la clave "error" en vez de propagar la excepción.
    """
    handler = _DISPATCH.get(name)
    if handler is None:
        return {"error": f"Tool desconocida: {name}"}
    try:
        return handler(tool_input)
    except Exception:
        logger.exception("Tool '%s' falló al ejecutarse.", name)
        return {"error": f"La tool '{name}' tuvo un problema al ejecutarse."}
