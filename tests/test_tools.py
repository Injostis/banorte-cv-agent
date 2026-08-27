import app.tools as tools_module
from app.tools import execute_tool


def test_get_summary_returns_nombre_titulo_resumen():
    result = execute_tool("get_summary", {})
    assert result["nombre"] == "Rodrigo Antonio Rios de los Santos"
    assert "titulo" in result
    assert "resumen" in result


def test_get_experience_returns_ryma_y_john_deere():
    result = execute_tool("get_experience", {})
    empresas = [job["empresa"] for job in result["experiencia"]]
    assert "Productos y Servicios RYMA S.A. de C.V." in empresas
    assert "John Deere" in empresas


def test_get_contact_no_incluye_telefono():
    result = execute_tool("get_contact", {})
    contacto = result["contacto"]
    assert contacto["email"] == "rod06@hotmail.es"
    assert "telefono" not in contacto


def test_get_projects_incluye_muralea():
    result = execute_tool("get_projects", {})
    nombres = [p["nombre"] for p in result["proyectos"]]
    assert "Muralea" in nombres


def test_get_ps_trophies_returns_total_and_list():
    result = execute_tool("get_ps_trophies", {})
    assert result["total_platinos"] == len(result["trofeos"])
    assert result["total_platinos"] > 0
    juegos = [t["juego"] for t in result["trofeos"]]
    assert any("Batman" in juego for juego in juegos)


def test_show_ps_trophies_table_returns_call_tool_result_with_resource():
    result = execute_tool("show_ps_trophies_table", {})
    types = [part["type"] for part in result["content"]]
    assert types == ["text", "resource"]
    assert result["content"][0]["text"]  # el fallback no está vacío
    assert result["content"][1]["resource"]["mimeType"] == "application/a2ui+json"


def test_unknown_tool_returns_error_dict_instead_of_raising():
    result = execute_tool("tool_que_no_existe", {})
    assert "error" in result


def test_tool_handler_exception_is_caught_not_propagated(monkeypatch):
    def _broken_handler(_):
        raise KeyError("clave_que_no_existe")

    monkeypatch.setitem(tools_module._DISPATCH, "get_summary", _broken_handler)

    result = execute_tool("get_summary", {})  # no debe levantar excepción

    assert "error" in result
    assert "get_summary" in result["error"]
