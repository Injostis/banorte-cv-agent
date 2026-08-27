"""Subconjunto propio del contrato Open Responses (openresponses.org).

No se valida el spec completo -- solo lo que este agente necesita: items de
tipo "message" en el input, y un output con items "function_call" (para que
las tool calls se vean en la UI de Banorte) seguido de un "message" final.
Deliberadamente tolerante en el parseo del input (acepta content como string
o como lista de partes) para no rechazar variaciones válidas del cliente por
un parseo demasiado estricto.
"""

import json
import time
import uuid
from typing import Any

from anthropic.types import MessageParam
from pydantic import BaseModel, ConfigDict

from app.a2ui import build_ps_trophies_messages, wrap_as_a2a_data_part
from app.agent import ToolCallRecord


class ResponsesRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str | None = None
    input: list[dict[str, Any]] | str


def _item_text(item: dict[str, Any]) -> str:
    """Extrae el texto plano de un item de tipo message, sin importar si su
    `content` viene como string o como lista de partes (input_text/output_text/text)."""
    content = item.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and "text" in part:
                parts.append(part["text"])
        return "\n".join(parts)
    return ""


def normalize_input(raw_input: list[dict[str, Any]] | str) -> list[dict[str, Any]]:
    """Normaliza el input de la request a una lista de items de tipo message."""
    if isinstance(raw_input, str):
        return [{"type": "message", "role": "user", "content": raw_input}]
    return raw_input


def to_anthropic_messages(items: list[dict[str, Any]]) -> list[MessageParam]:
    """Convierte los items de input (Open Responses) a mensajes formato Anthropic.

    Solo se traducen roles user/assistant -- un item de sistema/developer que
    mande el cliente (el campo "Instrucciones" opcional de Banorte) se
    ignora a propósito: el agente ya trae su propio system prompt completo
    (ver app/agent.py) y no depende de que un tercero se lo complemente
    correctamente en cada request.
    """
    messages: list[MessageParam] = []
    for item in items:
        if item.get("type") != "message":
            continue
        role = item.get("role")
        text = _item_text(item)
        if not text:
            continue
        if role == "user":
            messages.append({"role": "user", "content": text})
        elif role == "assistant":
            messages.append({"role": "assistant", "content": text})
    return messages


def last_user_text(items: list[dict[str, Any]]) -> str:
    for item in reversed(items):
        if item.get("type") == "message" and item.get("role") == "user":
            return _item_text(item)
    return ""


def transcript_before_last_user(items: list[dict[str, Any]]) -> str:
    """Arma un transcript en texto plano de la conversación previa (sin el
    último mensaje del usuario), para dárselo como contexto al guardrail."""
    messages = to_anthropic_messages(items)
    if messages and messages[-1]["role"] == "user":
        messages = messages[:-1]
    lines = []
    for message in messages:
        speaker = "Usuario" if message["role"] == "user" else "Agente"
        lines.append(f"{speaker}: {message['content']}")
    return "\n".join(lines)


def build_response(*, model: str, final_text: str, tool_calls: list[ToolCallRecord]) -> dict[str, Any]:
    output: list[dict[str, Any]] = []
    for call in tool_calls:
        output.append(
            {
                "id": f"fc_{uuid.uuid4().hex[:24]}",
                "type": "function_call",
                "call_id": f"call_{uuid.uuid4().hex[:24]}",
                "name": call.name,
                "arguments": json.dumps(call.input, ensure_ascii=False),
                "status": "completed",
            }
        )
    output.append(
        {
            "id": f"msg_{uuid.uuid4().hex[:24]}",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": final_text, "annotations": []}],
        }
    )

    # La parte de A2UI va como su propio item de "output", al mismo nivel
    # que "message" y "function_call" -- no anidada dentro del content del
    # mensaje. Open Responses discrimina items por "type" y A2A discrimina
    # Parts por "kind"; mezclar ambos esquemas dentro del mismo array de
    # content era una hipótesis real de por qué no se detectaba.
    ps_trophies_call = next((call for call in tool_calls if call.name == "get_ps_trophies"), None)
    if ps_trophies_call is not None:
        trofeos = ps_trophies_call.output.get("trofeos")
        if isinstance(trofeos, list) and trofeos:
            messages = build_ps_trophies_messages(trofeos)
            output.append(wrap_as_a2a_data_part(messages))

    return {
        "id": f"resp_{uuid.uuid4().hex[:24]}",
        "object": "response",
        "created_at": int(time.time()),
        "model": model,
        "status": "completed",
        "output": output,
    }
