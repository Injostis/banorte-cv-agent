from app.images import extract_image_urls, image_block_from_url

# 1x1 PNG transparente, el clásico de pruebas.
_TINY_PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="


def test_image_block_from_data_uri():
    url = f"data:image/png;base64,{_TINY_PNG_B64}"
    block = image_block_from_url(url)
    assert block == {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": _TINY_PNG_B64},
    }


def test_image_block_from_https_url():
    block = image_block_from_url("https://example.com/perro.jpg")
    assert block == {"type": "image", "source": {"type": "url", "url": "https://example.com/perro.jpg"}}


def test_image_block_rejects_unsupported_media_type():
    url = f"data:image/svg+xml;base64,{_TINY_PNG_B64}"
    assert image_block_from_url(url) is None


def test_image_block_rejects_garbage():
    assert image_block_from_url("no es una url de imagen") is None


def test_extract_image_urls_string_form():
    item = {"content": [{"type": "input_image", "image_url": "https://example.com/a.png"}]}
    assert extract_image_urls(item) == ["https://example.com/a.png"]


def test_extract_image_urls_nested_object_form():
    item = {"content": [{"type": "input_image", "image_url": {"url": "https://example.com/b.png"}}]}
    assert extract_image_urls(item) == ["https://example.com/b.png"]


def test_extract_image_urls_no_images_returns_empty():
    item = {"content": "solo texto plano"}
    assert extract_image_urls(item) == []
