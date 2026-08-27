import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from langfuse import get_client

from app.agent import run_agent
from app.config import settings
from app.guardrails import GuardrailRejection, check_input
from app.observability import init_langfuse
from app.responses_schema import (
    ResponsesRequest,
    build_response,
    last_user_has_image,
    last_user_text,
    normalize_input,
    to_anthropic_messages,
    transcript_before_last_user,
)

logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("banorte_cv_agent")

init_langfuse()


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    yield
    # Las trazas de Langfuse se mandan en lote, no al instante -- sin este
    # flush, las últimas quedarían en el buffer y se perderían cuando
    # Render reinicia o redespliega la instancia.
    get_client().flush()


app = FastAPI(
    title="Agente de CV — Rodrigo Rios",
    description="Agente conversacional sobre la trayectoria profesional de Rodrigo, compatible con Open Responses.",
    version="0.1.0",
    lifespan=lifespan,
)

bearer_scheme = HTTPBearer(auto_error=False)


def verify_token(credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme)) -> None:
    if credentials is None or credentials.credentials != settings.agent_bearer_token:
        raise HTTPException(status_code=401, detail="Token inválido o ausente.")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/.well-known/agent-card.json")
def agent_card(request: Request) -> dict[str, Any]:
    """Tarjeta de agente (protocolo A2A). Declara el soporte experimental de
    A2UI vía capabilities.extensions -- sin esta declaración, un cliente
    que sí sabe renderizar A2UI podría no buscarlo siquiera en las
    respuestas. Pública a propósito (sin auth): así es como funciona el
    descubrimiento de agentes."""
    base_url = str(request.base_url).rstrip("/")
    return {
        "protocolVersion": "0.3.0",
        "name": "Agente de CV — Rodrigo Rios",
        "description": (
            "Conversa sobre la trayectoria profesional de Rodrigo: experiencia, proyectos, habilidades y educación."
        ),
        "url": f"{base_url}/responses",
        "supportedInterfaces": [
            {
                "url": f"{base_url}/responses",
                "protocolBinding": "HTTP+JSON",
                "protocolVersion": "0.3.0",
            }
        ],
        "version": "0.1.0",
        "capabilities": {
            "streaming": False,
            "pushNotifications": False,
            "extensions": [
                {
                    "uri": "https://a2ui.org/a2a-extension/a2ui/v1.0",
                    "description": "Puede incluir componentes A2UI en sus respuestas, siempre con respaldo de texto.",
                    "required": True,
                }
            ],
        },
        "defaultInputModes": ["text/plain"],
        "defaultOutputModes": ["text/plain", "application/json+a2ui"],
        "skills": [
            {
                "id": "cv-qa",
                "name": "Preguntas sobre el perfil profesional",
                "description": (
                    "Responde preguntas sobre experiencia laboral, proyectos, habilidades y educación de Rodrigo."
                ),
                "tags": ["cv", "recruiting"],
                "examples": ["¿Qué proyectos has construido?", "Cuéntame de tu experiencia laboral"],
            }
        ],
    }


@app.post("/responses", dependencies=[Depends(verify_token)])
def create_response(request: ResponsesRequest) -> dict[str, Any]:
    start = time.monotonic()
    items = normalize_input(request.input)
    user_text = last_user_text(items)
    has_image = last_user_has_image(items)

    if not user_text and not has_image:
        raise HTTPException(status_code=400, detail="No se encontró un mensaje de usuario en 'input'.")

    # El guardrail de texto no tiene nada que clasificar en un mensaje que
    # es solo una imagen sin texto -- se salta, y la seguridad para ese
    # caso recae en la regla del system prompt de nunca seguir
    # instrucciones encontradas dentro de una imagen (ver app/agent.py).
    if user_text:
        context = transcript_before_last_user(items)
        try:
            check_input(user_text, context=context)
        except GuardrailRejection as rejection:
            logger.info("Guardrail rechazó un mensaje. Motivo interno: %s", rejection.internal_reason)
            return build_response(model=settings.claude_model, final_text=rejection.user_message, tool_calls=[])

    messages = to_anthropic_messages(items)

    try:
        result = run_agent(messages)
    except Exception:
        # Red de seguridad general: un timeout/rate-limit/error transitorio de
        # la API de Claude, o cualquier otro fallo inesperado, nunca debe
        # llegar a Banorte como un 500 crudo -- el chat debe poder mostrar
        # algo coherente y el usuario debe poder reintentar.
        logger.exception("Fallo inesperado ejecutando el agente.")
        return build_response(
            model=settings.claude_model,
            final_text="Tuve un problema técnico procesando tu pregunta. ¿Puedes intentar de nuevo?",
            tool_calls=[],
        )

    elapsed_ms = (time.monotonic() - start) * 1000
    tool_names = [call.name for call in result.tool_calls]
    logger.info("Respuesta generada en %.0fms. Tools usadas: %s", elapsed_ms, tool_names or "ninguna")

    return build_response(model=settings.claude_model, final_text=result.final_text, tool_calls=result.tool_calls)
