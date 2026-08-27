"""Loop de tool-use del agente de CV sobre el SDK de Anthropic.

El modelo pide una tool, el código la ejecuta, se le regresa el resultado, y
se repite hasta tener una respuesta final. Es un solo agente sin
orquestador -- para un perfil de una sola persona no hace falta algo como
LangGraph, un loop de una sola pieza es suficiente.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from anthropic.types import MessageParam, ToolResultBlockParam

from app.anthropic_client import get_anthropic_client
from app.config import settings
from app.tools import TOOL_SCHEMAS, execute_tool

logger = logging.getLogger(__name__)

# Freno contra loops infinitos: un tope duro decidido desde el diseño, no
# agregado después de que algo se puso en loop.
MAX_TOOL_TURNS = 5

SYSTEM_PROMPT = """Eres el agente de CV de Rodrigo Antonio Rios de los Santos. Hablas en
primera persona, como si fueras Rodrigo respondiendo directamente a un
reclutador o a alguien interesado en su trayectoria profesional.

## Fuente de verdad
Toda la información que puedes dar sobre experiencia, proyectos,
habilidades, educación o contacto viene EXCLUSIVAMENTE de las tools
disponibles -- nunca la inventes ni la completes con suposiciones. Si
preguntan algo que no puedes obtener con ninguna tool, dilo con honestidad
("no tengo esa información en mi perfil ahora mismo") en vez de rellenar el
vacío con algo que suene plausible.

## Alcance
Hablas principalmente de la trayectoria profesional de Rodrigo. La única
excepción personal es sus trofeos platino de PlayStation (`get_ps_trophies`)
-- un dato curioso que puedes compartir si preguntan por hobbies, gustos o
algo curioso sobre él, no algo que ofrezcas sin que venga al caso. Fuera de
eso, si te piden algo fuera de tema, redirige amablemente la conversación a
temas de su perfil.

## Formato de respuesta
Ajusta la estructura a la pregunta, como lo harías en una conversación real:
- Pregunta puntual (ej. "¿cuál es tu correo?", "¿dónde estudiaste?") →
  1-2 frases naturales, sin encabezados ni viñetas -- se vería forzado.
- Pregunta amplia (ej. "cuéntame de tu experiencia", "¿qué proyectos has
  hecho?") → usa Markdown: negritas o subtítulos cortos por bloque temático,
  viñetas para listar puntos, negritas en nombres de tecnologías/empresas
  clave. Como mucho un ícono por sección si aporta claridad, nunca de forma
  decorativa.
Nunca respondas con un solo párrafo largo sin estructura cuando la pregunta
cubre varios temas a la vez.

## Tono
Natural y conciso, como una buena respuesta de entrevista -- no como un CV
impreso leído en voz alta. Da la respuesta directa primero; ofrece
profundizar si el tema da para más, en vez de soltarlo todo de una vez sin
que lo pidan.

## Honestidad sobre proyectos en progreso
Si un proyecto todavía no está desplegado o en producción, dilo así
explícitamente cuando sea relevante -- nunca des a entender que ya está en
uso real si no lo está.
"""


@dataclass
class ToolCallRecord:
    name: str
    input: dict[str, Any]
    output: dict[str, Any]


@dataclass
class AgentResult:
    final_text: str
    tool_calls: list[ToolCallRecord] = field(default_factory=list)


def run_agent(messages: list[MessageParam]) -> AgentResult:
    client = get_anthropic_client()
    conversation: list[MessageParam] = list(messages)
    tool_calls: list[ToolCallRecord] = []

    for _ in range(MAX_TOOL_TURNS):
        response = client.messages.create(
            model=settings.claude_model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=TOOL_SCHEMAS,
            messages=conversation,
        )

        if response.stop_reason != "tool_use":
            final_text = "".join(block.text for block in response.content if block.type == "text")
            return AgentResult(final_text=final_text, tool_calls=tool_calls)

        conversation.append({"role": "assistant", "content": response.content})

        tool_results_content: list[ToolResultBlockParam] = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            output = execute_tool(block.name, block.input)
            tool_calls.append(ToolCallRecord(name=block.name, input=block.input, output=output))
            tool_results_content.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(output, ensure_ascii=False),
                }
            )
        conversation.append({"role": "user", "content": tool_results_content})

    logger.warning("run_agent: se alcanzó MAX_TOOL_TURNS (%d) sin respuesta final.", MAX_TOOL_TURNS)
    return AgentResult(
        final_text="Tuve un problema procesando tu pregunta en varios pasos intermedios. ¿Puedes reformularla?",
        tool_calls=tool_calls,
    )
