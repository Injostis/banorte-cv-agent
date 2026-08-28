"""Soporte de imágenes en la entrada del usuario.

Convierte una URL de imagen (URL normal o data URI en base64) al bloque de
imagen que espera el SDK de Anthropic. Acepta item.image_url tanto como
string como en forma de {"url": ...}.
"""

import re
from typing import Any, Literal, cast, get_args

from anthropic.types import ImageBlockParam

_DATA_URI_RE = re.compile(r"^data:(?P<media_type>[\w/+-]+);base64,(?P<data>.+)$", re.DOTALL)

_AllowedMediaType = Literal["image/jpeg", "image/png", "image/gif", "image/webp"]
_ALLOWED_MEDIA_TYPES: tuple[str, ...] = get_args(_AllowedMediaType)


def image_block_from_url(url: str) -> ImageBlockParam | None:
    """Convierte una URL de imagen a un ImageBlockParam de Anthropic.

    Regresa None si el formato no es reconocido o el media_type no está
    permitido."""
    match = _DATA_URI_RE.match(url)
    if match:
        media_type = match.group("media_type")
        if media_type not in _ALLOWED_MEDIA_TYPES:
            return None
        allowed_media_type = cast(_AllowedMediaType, media_type)
        block: ImageBlockParam = {
            "type": "image",
            "source": {"type": "base64", "media_type": allowed_media_type, "data": match.group("data")},
        }
        return block
    if url.startswith(("http://", "https://")):
        url_block: ImageBlockParam = {"type": "image", "source": {"type": "url", "url": url}}
        return url_block
    return None


def extract_image_urls(item: dict[str, Any]) -> list[str]:
    """Saca las URLs de imagen de un item de input, tolerando variaciones
    de formato: item.image_url como string, o como {"url": "..."}."""
    content = item.get("content", "")
    if not isinstance(content, list):
        return []

    urls: list[str] = []
    for part in content:
        if not isinstance(part, dict) or part.get("type") not in ("input_image", "image"):
            continue
        image_url = part.get("image_url")
        if isinstance(image_url, str):
            urls.append(image_url)
        elif isinstance(image_url, dict) and isinstance(image_url.get("url"), str):
            urls.append(image_url["url"])
        elif isinstance(part.get("url"), str):
            urls.append(part["url"])
    return urls
