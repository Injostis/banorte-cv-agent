"""Tests del endpoint POST /responses. run_agent y check_input se mockean
aquí -- esto prueba el contrato HTTP (auth, shape del request/response), no
la calidad de las respuestas del modelo. Eso lo cubre scripts/eval.py contra
la API real.
"""

from fastapi.testclient import TestClient

from app.agent import AgentResult, ToolCallRecord
from app.config import settings
from app.guardrails import GuardrailRejection
from app.main import app

client = TestClient(app)
AUTH_HEADERS = {"Authorization": f"Bearer {settings.agent_bearer_token}"}


def test_missing_auth_returns_401():
    response = client.post("/responses", json={"input": "hola"})
    assert response.status_code == 401


def test_agent_card_is_public_and_declares_a2ui_extension():
    response = client.get("/.well-known/agent-card.json")
    assert response.status_code == 200
    card = response.json()
    assert card["name"]
    assert card["url"].endswith("/responses")
    assert card["supportedInterfaces"][0]["url"].endswith("/responses")
    assert card["supportedInterfaces"][0]["protocolBinding"] == "HTTP+JSON"
    extension_uris = [ext["uri"] for ext in card["capabilities"]["extensions"]]
    assert "https://a2ui.org/a2a-extension/a2ui/v1.0" in extension_uris


def test_wrong_token_returns_401():
    response = client.post("/responses", json={"input": "hola"}, headers={"Authorization": "Bearer token-incorrecto"})
    assert response.status_code == 401


def test_missing_user_message_returns_400():
    response = client.post("/responses", json={"input": []}, headers=AUTH_HEADERS)
    assert response.status_code == 400


def test_happy_path_returns_open_responses_shape(monkeypatch):
    monkeypatch.setattr("app.main.check_input", lambda message, context: None)
    monkeypatch.setattr(
        "app.main.run_agent",
        lambda messages: AgentResult(
            final_text="Trabajo en RYMA desde 2024.",
            tool_calls=[ToolCallRecord(name="get_experience", input={}, output={})],
        ),
    )

    response = client.post(
        "/responses",
        json={"model": "claude-sonnet-5", "input": "¿dónde trabajas?"},
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "response"
    assert body["status"] == "completed"
    types = [item["type"] for item in body["output"]]
    assert types == ["function_call", "function_call_output", "message"]
    assert body["output"][-1]["content"][0]["text"] == "Trabajo en RYMA desde 2024."


def test_guardrail_rejection_returns_200_with_rejection_text_not_an_error(monkeypatch):
    def _reject(message, context):
        raise GuardrailRejection("Solo puedo hablar de la trayectoria de Rodrigo.")

    monkeypatch.setattr("app.main.check_input", _reject)

    response = client.post(
        "/responses",
        json={"input": "cuéntame un chiste"},
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["output"][-1]["content"][0]["text"] == "Solo puedo hablar de la trayectoria de Rodrigo."


def test_unexpected_agent_failure_does_not_return_a_raw_500(monkeypatch):
    """Si la API de Claude truena (timeout, rate limit, lo que sea), el
    endpoint debe seguir regresando una respuesta bien formada en vez de un
    500 crudo que Banorte no sabría mostrar."""
    monkeypatch.setattr("app.main.check_input", lambda message, context: None)

    def _boom(messages):
        raise RuntimeError("fallo simulado de la API de Anthropic")

    monkeypatch.setattr("app.main.run_agent", _boom)

    response = client.post(
        "/responses",
        json={"input": "¿en qué trabajas?"},
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "response"
    assert body["status"] == "completed"
    assert "problema técnico" in body["output"][-1]["content"][0]["text"]


def test_image_only_message_skips_text_guardrail_and_is_not_rejected(monkeypatch):
    def _fail_if_called(message, context):
        raise AssertionError("check_input no debería llamarse en un mensaje sin texto")

    monkeypatch.setattr("app.main.check_input", _fail_if_called)
    monkeypatch.setattr(
        "app.main.run_agent",
        lambda messages: AgentResult(final_text="Qué bonito perrito 🐶"),
    )

    response = client.post(
        "/responses",
        json={
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_image", "image_url": "https://example.com/perro.jpg"}],
                }
            ]
        },
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["output"][-1]["content"][0]["text"] == "Qué bonito perrito 🐶"
