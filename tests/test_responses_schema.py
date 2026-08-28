import json

from app.a2ui import build_ps_trophies_tool_result
from app.agent import ToolCallRecord
from app.responses_schema import (
    build_response,
    last_user_has_image,
    last_user_text,
    normalize_input,
    stream_response_events,
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
    assert types == ["function_call", "function_call_output", "message"]
    assert response["output"][0]["name"] == "get_summary"
    assert response["output"][-1]["content"][0]["text"] == "Soy Rodrigo."


def test_build_response_function_call_output_shares_call_id_and_carries_result():
    tool_calls = [ToolCallRecord(name="get_summary", input={}, output={"nombre": "Rodrigo"})]
    response = build_response(model="claude-sonnet-5", final_text="Soy Rodrigo.", tool_calls=tool_calls)

    call_item, output_item = response["output"][0], response["output"][1]
    assert output_item["call_id"] == call_item["call_id"]
    # "output" va como string JSON, no como objeto anidado.
    assert isinstance(output_item["output"], str)
    assert json.loads(output_item["output"]) == {"nombre": "Rodrigo"}


def test_build_response_adds_a2ui_resource_part_for_ps_trophies():
    trofeos = [
        {"juego": "Grounded", "plataforma": "PS5", "nombre_trofeo": "X", "porcentaje_jugadores_con_este_trofeo": "0.5"}
    ]
    tool_result = build_ps_trophies_tool_result(trofeos)
    tool_calls = [ToolCallRecord(name="show_ps_trophies_table", input={}, output=tool_result)]

    response = build_response(model="claude-sonnet-5", final_text="Tengo varios platinos.", tool_calls=tool_calls)

    types = [item["type"] for item in response["output"]]
    assert types == ["function_call", "function_call_output", "message"]

    message_item = response["output"][2]
    assert message_item["content"][0]["text"] == "Tengo varios platinos."

    output_item = response["output"][1]
    assert output_item["call_id"] == response["output"][0]["call_id"]
    resource_part = json.loads(output_item["output"])["content"][1]
    assert resource_part["type"] == "resource"
    assert resource_part["resource"]["mimeType"] == "application/a2ui+json"
    assert resource_part["resource"]["uri"].startswith("a2ui://banorte-cv-agent/ps_trophies")
    a2ui_messages = json.loads(resource_part["resource"]["text"])
    surface_id = a2ui_messages[0]["createSurface"]["surfaceId"]
    assert surface_id.startswith("ps_trophies_")
    assert a2ui_messages[1]["updateComponents"]["surfaceId"] == surface_id
    assert a2ui_messages[2]["updateDataModel"]["surfaceId"] == surface_id


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


def test_stream_response_events_matches_build_response_content():
    tool_calls = [ToolCallRecord(name="get_summary", input={}, output={"nombre": "Rodrigo"})]
    events = list(stream_response_events(model="claude-sonnet-5", final_text="Soy Rodrigo.", tool_calls=tool_calls))

    assert events[-1] == "data: [DONE]\n\n"
    assert all(e.startswith("data: ") and e.endswith("\n\n") for e in events)

    event_types = [json.loads(e[len("data: ") :])["type"] for e in events[:-1]]
    assert event_types[0] == "response.created"
    assert event_types[-1] == "response.completed"
    assert "response.output_item.added" in event_types
    assert "response.output_item.done" in event_types
    assert "response.output_text.delta" in event_types

    completed = json.loads(events[-2][len("data: ") :])
    assert completed["response"]["status"] == "completed"
    types_in_output = [item["type"] for item in completed["response"]["output"]]
    assert types_in_output == ["function_call", "function_call_output", "message"]
    assert completed["response"]["output"][-1]["content"][0]["text"] == "Soy Rodrigo."


def test_stream_response_events_includes_a2ui_resource_item():
    trofeos = [{"juego": "Grounded", "porcentaje_jugadores_con_este_trofeo": "0.5"}]
    tool_result = build_ps_trophies_tool_result(trofeos)
    tool_calls = [ToolCallRecord(name="show_ps_trophies_table", input={}, output=tool_result)]

    events = list(stream_response_events(model="claude-sonnet-5", final_text="Aquí están.", tool_calls=tool_calls))
    completed = json.loads(events[-2][len("data: ") :])
    output_item = next(item for item in completed["response"]["output"] if item["type"] == "function_call_output")
    result_content_types = [part["type"] for part in json.loads(output_item["output"])["content"]]
    assert "resource" in result_content_types
