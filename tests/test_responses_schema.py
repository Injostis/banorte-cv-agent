from app.agent import ToolCallRecord
from app.responses_schema import (
    build_response,
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
