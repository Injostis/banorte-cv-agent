import json

from app.agent import ToolCallRecord
from app.responses_schema import (
    build_response,
    last_user_has_image,
    last_user_text,
    normalize_input,
    to_anthropic_messages,
    transcript_before_last_user,
)


def test_normalize_input_accepts_plain_string():
    items = normalize_input("hola")
    assert items == [{"type": "message", "role": "user", "content": "hola"}]


def test_normalize_input_passes_through_list():
    raw = [{"type": "message", "role": "user", "content": "hola"}]
    assert normalize_input(raw) == raw


def test_to_anthropic_messages_ignores_system_role():
    items = [
        {"type": "message", "role": "system", "content": "instrucciones del cliente"},
        {"type": "message", "role": "user", "content": "hola"},
    ]
    messages = to_anthropic_messages(items)
    assert messages == [{"role": "user", "content": "hola"}]


def test_to_anthropic_messages_handles_content_as_parts_list():
    items = [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hola"}]}]
    assert to_anthropic_messages(items) == [{"role": "user", "content": "hola"}]


def test_last_user_text_picks_most_recent_user_message():
    items = [
        {"type": "message", "role": "user", "content": "primera pregunta"},
        {"type": "message", "role": "assistant", "content": "primera respuesta"},
        {"type": "message", "role": "user", "content": "segunda pregunta"},
    ]
    assert last_user_text(items) == "segunda pregunta"


def test_transcript_before_last_user_excludes_final_message():
    items = [
        {"type": "message", "role": "user", "content": "primera pregunta"},
        {"type": "message", "role": "assistant", "content": "primera respuesta"},
        {"type": "message", "role": "user", "content": "segunda pregunta (editada)"},
    ]
    transcript = transcript_before_last_user(items)
    assert "primera pregunta" in transcript
    assert "primera respuesta" in transcript
    assert "segunda pregunta" not in transcript


def test_build_response_includes_function_call_items_before_message():
    tool_calls = [ToolCallRecord(name="get_summary", input={}, output={"nombre": "Rodrigo"})]
    response = build_response(model="claude-sonnet-5", final_text="Soy Rodrigo.", tool_calls=tool_calls)

    assert response["object"] == "response"
    assert response["status"] == "completed"
    types = [item["type"] for item in response["output"]]
    assert types == ["function_call", "message"]
    assert response["output"][0]["name"] == "get_summary"
    assert response["output"][-1]["content"][0]["text"] == "Soy Rodrigo."


def test_build_response_adds_a2ui_resource_part_for_ps_trophies():
    trofeos = [
        {"juego": "Grounded", "plataforma": "PS5", "nombre_trofeo": "X", "porcentaje_jugadores_con_este_trofeo": "0.5"}
    ]
    tool_calls = [ToolCallRecord(name="get_ps_trophies", input={}, output={"trofeos": trofeos})]

    response = build_response(model="claude-sonnet-5", final_text="Tengo varios platinos.", tool_calls=tool_calls)

    # El "resource" de A2UI va como su propio item de "output", después del
    # mensaje de texto -- no anidado dentro de su content.
    types = [item["type"] for item in response["output"]]
    assert types == ["function_call", "message", "resource"]

    message_item = response["output"][1]
    assert message_item["content"][0]["text"] == "Tengo varios platinos."

    resource_item = response["output"][2]
    assert resource_item["resource"]["mimeType"] == "application/a2ui+json"
    assert resource_item["resource"]["uri"] == "a2ui://banorte-cv-agent/ps_trophies"
    a2ui_messages = json.loads(resource_item["resource"]["text"])
    assert a2ui_messages[0]["createSurface"]["surfaceId"] == "ps_trophies"
    assert a2ui_messages[1]["updateComponents"]["surfaceId"] == "ps_trophies"
    assert a2ui_messages[2]["updateDataModel"]["surfaceId"] == "ps_trophies"


def test_build_response_no_a2ui_part_without_ps_trophies_tool():
    tool_calls = [ToolCallRecord(name="get_summary", input={}, output={"nombre": "Rodrigo"})]
    response = build_response(model="claude-sonnet-5", final_text="Soy Rodrigo.", tool_calls=tool_calls)

    content = response["output"][-1]["content"]
    assert len(content) == 1


def test_to_anthropic_messages_builds_multipart_content_for_image_plus_text():
    items = [
        {
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_image", "image_url": "https://example.com/perro.jpg"},
                {"type": "input_text", "text": "¿qué te parece?"},
            ],
        }
    ]
    messages = to_anthropic_messages(items)
    assert len(messages) == 1
    content = messages[0]["content"]
    assert content[0] == {"type": "image", "source": {"type": "url", "url": "https://example.com/perro.jpg"}}
    assert content[1] == {"type": "text", "text": "¿qué te parece?"}


def test_to_anthropic_messages_image_only_no_text():
    items = [
        {"type": "message", "role": "user", "content": [{"type": "input_image", "image_url": "https://x.com/a.jpg"}]}
    ]
    messages = to_anthropic_messages(items)
    assert len(messages) == 1
    assert messages[0]["content"] == [{"type": "image", "source": {"type": "url", "url": "https://x.com/a.jpg"}}]


def test_last_user_has_image_true_and_false():
    with_image = [
        {"type": "message", "role": "user", "content": [{"type": "input_image", "image_url": "https://x.com/a.jpg"}]}
    ]
    without_image = [{"type": "message", "role": "user", "content": "hola"}]
    assert last_user_has_image(with_image) is True
    assert last_user_has_image(without_image) is False
