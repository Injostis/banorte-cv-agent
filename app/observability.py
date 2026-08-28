"""Observabilidad con Langfuse.

Si no hay llaves configuradas, el cliente queda deshabilitado
(tracing_enabled=False) y no se generan trazas.
"""

import logging
from functools import lru_cache

from anthropic.types import Message
from langfuse import Langfuse, get_client

from app.config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def init_langfuse() -> Langfuse:
    """Crea y regresa el cliente de Langfuse, cacheado tras la primera llamada."""
    configured = bool(settings.langfuse_public_key and settings.langfuse_secret_key)
    if not configured:
        logging.getLogger("langfuse").setLevel(logging.ERROR)
    return Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
        tracing_enabled=configured,
    )


def record_usage(response: Message, model: str) -> None:
    """Registra tokens de entrada/salida en la generación actual de Langfuse."""
    try:
        get_client().update_current_generation(
            model=model,
            usage_details={"input": response.usage.input_tokens, "output": response.usage.output_tokens},
        )
    except Exception:
        logger.warning("No se pudo registrar uso de tokens en Langfuse.", exc_info=True)
