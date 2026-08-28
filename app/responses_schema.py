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
from collections.abc import Iterator
from typing import Any

from anthropic.types import ImageBlockParam, MessageParam, TextBlockParam
from pydantic import BaseModel, ConfigDict

from app.agent import ToolCallRecord
from app.images import extract_image_urls, image_block_from_url


class ResponsesRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str | None = None
    input: list[dict[str, Any]] | str
    stream: bool = False


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

    Si un mensaje de usuario trae imágenes, se arma como contenido
    multi-parte (imagen(es) + texto si lo hay) en vez de solo texto plano.
    """
    messages: list[MessageParam] = []
    for item in items:
        if item.get("type") != "message":
            continue
        role = item.get("role")
        text = _item_text(item)

        if role == "user":
            image_blocks: list[ImageBlockParam] = [
                block for url in extract_image_urls(item) if (block := image_block_from_url(url)) is not None
            ]
            if image_blocks:
                content_parts: list[ImageBlockParam | TextBlockParam] = list(image_blocks)
                if text:
                    content_parts.append({"type": "text", "text": text})
                messages.append({"role": "user", "content": content_parts})
                continue
            if text:
                messages.append({"role": "user", "content": text})
        elif role == "assistant" and text:
            messages.append({"role": "assistant", "content": text})
    return messages


def last_user_text(items: list[dict[str, Any]]) -> str:
    for item in reversed(items):
        if item.get("type") == "message" and item.get("role") == "user":
            return _item_text(item)
    return ""


def last_user_has_image(items: list[dict[str, Any]]) -> bool:
    for item in reversed(items):
        if item.get("type") == "message" and item.get("role") == "user":
            return bool(extract_image_urls(item))
    return False


def transcript_before_last_user(items: list[dict[str, Any]]) -> str:
    """Arma un transcript en texto plano de la conversación previa (sin el
    último mensaje del usuario), para dárselo como contexto al guardrail."""
    messages = to_anthropic_messages(items)
    if messages and messages[-1]["role"] == "user":
        messages = messages[:-1]
    lines = []
    for message in messages:
        speaker = "Usuario" if message["role"] == "user" else "Agente"
        content = message["content"] if isinstance(message["content"], str) else "[imagen]"
        lines.append(f"{speaker}: {content}")
    return "\n".join(lines)


def _build_output_items(final_text: str, tool_calls: list[ToolCallRecord]) -> list[dict[str, Any]]:
    """Arma la lista de items de "output" -- compartida entre la respuesta
    completa (build_response) y la respuesta en streaming
    (stream_response_events), para no duplicar esta lógica dos veces."""
    output: list[dict[str, Any]] = []
    for call in tool_calls:
        call_id = f"call_{uuid.uuid4().hex[:24]}"
        output.append(
            {
                "id": f"fc_{uuid.uuid4().hex[:24]}",
                "type": "function_call",
                "call_id": call_id,
                "name": call.name,
                "arguments": json.dumps(call.input, ensure_ascii=False),
                "status": "completed",
            }
        )

        # El resultado va en su propio item, referenciando el mismo call_id
        # -- así el cliente asocia la llamada con su resultado y sabe que ya
        # terminó. Si la tool regresó un CallToolResult (una superficie
        # A2UI, ver app/a2ui.py), su content part "resource" (convención
        # EmbeddedResource de MCP) viaja aquí dentro. `output` va como
        # string JSON, no como objeto anidado.
        output.append(
            {
                "id": f"fco_{uuid.uuid4().hex[:24]}",
                "type": "function_call_output",
                "call_id": call_id,
                "output": json.dumps(call.output, ensure_ascii=False),
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
    return output


def build_response(*, model: str, final_text: str, tool_calls: list[ToolCallRecord]) -> dict[str, Any]:
    return {
        "id": f"resp_{uuid.uuid4().hex[:24]}",
        "object": "response",
        "created_at": int(time.time()),
        "model": model,
        "status": "completed",
        "output": _build_output_items(final_text, tool_calls),
    }


def stream_response_events(*, model: str, final_text: str, tool_calls: list[ToolCallRecord]) -> Iterator[str]:
    """Genera la misma respuesta que build_response(), como una secuencia
    de eventos SSE en vez de un JSON completo de una sola pieza. No es
    streaming token por token real -- la respuesta ya está calculada
    completa antes de emitirse como eventos."""
    response_id = f"resp_{uuid.uuid4().hex[:24]}"
    created_at = int(time.time())
    output_items = _build_output_items(final_text, tool_calls)

    def _event(event_type: str, data: dict[str, Any]) -> str:
        payload = {"type": event_type, **data}
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    yield _event(
        "response.created",
        {
            "response": {
                "id": response_id,
                "object": "response",
                "created_at": created_at,
                "model": model,
                "status": "in_progress",
                "output": [],
            }
        },
    )

    for index, item in enumerate(output_items):
        yield _event("response.output_item.added", {"output_index": index, "item": item})
        if item["type"] == "message":
            text = item["content"][0]["text"]
            yield _event("response.output_text.delta", {"output_index": index, "content_index": 0, "delta": text})
            yield _event("response.output_text.done", {"output_index": index, "content_index": 0, "text": text})
        yield _event("response.output_item.done", {"output_index": index, "item": item})

    final_response = {
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "model": model,
        "status": "completed",
        "output": output_items,
    }
    yield _event("response.completed", {"response": final_response})
    yield "data: [DONE]\n\n"
