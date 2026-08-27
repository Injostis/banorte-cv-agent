"""Observabilidad con Langfuse.

Opcional a propósito: si no hay llaves configuradas, el cliente queda
deshabilitado (tracing_enabled=False) y el agente sigue funcionando
exactamente igual, sin trazas -- Langfuse nunca debe ser un punto de falla
para la conversación real. Por eso record_usage() atrapa cualquier error
(red, cliente mal configurado) y solo lo loguea.
"""

import logging
from functools import lru_cache

from anthropic.types import Message
from langfuse import Langfuse, get_client

from app.config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def init_langfuse() -> Langfuse:
    """Crea el cliente de Langfuse una sola vez, al arrancar la app. Los
    decoradores @observe() usan este mismo cliente por dentro (patrón
    singleton de la librería), sin que haya que pasarlo explícitamente."""
    configured = bool(settings.langfuse_public_key and settings.langfuse_secret_key)
    if not configured:
        # Sin llaves, el SDK avisa "disabled" en cada operación -- ya lo
        # sabemos, no hace falta que inunde los logs en cada request.
        logging.getLogger("langfuse").setLevel(logging.ERROR)
    return Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
        tracing_enabled=configured,
    )


def record_usage(response: Message, model: str) -> None:
    """Registra tokens de entrada/salida en la generación actual de
    Langfuse (para seguimiento de costos). Si falla por lo que sea, solo
    se loguea -- nunca debe tumbar la respuesta real al usuario."""
    try:
        get_client().update_current_generation(
            model=model,
            usage_details={"input": response.usage.input_tokens, "output": response.usage.output_tokens},
        )
    except Exception:
        logger.warning("No se pudo registrar uso de tokens en Langfuse.", exc_info=True)
