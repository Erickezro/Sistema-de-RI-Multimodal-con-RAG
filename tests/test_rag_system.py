"""Tests unitarios para rag_system.py (pipeline RAG / helpers)."""

from types import SimpleNamespace

import pytest

import rag_system


def _make_response(content):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


class _FakeClient:
    def __init__(self, behaviors):
        self._behaviors = behaviors
        self.calls = []
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    def _create(self, model, messages, temperature, max_tokens):
        self.calls.append((model, messages))
        behavior = self._behaviors[model]
        if isinstance(behavior, Exception):
            raise behavior
        return _make_response(behavior)


# --------------------------- resolve_image_url --------------------------

def test_resolve_image_url_none():
    assert rag_system.resolve_image_url(None) is None


def test_resolve_image_url_http_passthrough():
    url = "http://example.com/a.jpg"
    assert rag_system.resolve_image_url(url) == url


def test_resolve_image_url_https_passthrough():
    url = "https://example.com/a.jpg"
    assert rag_system.resolve_image_url(url) == url


def test_resolve_image_url_local_missing_returns_none():
    assert rag_system.resolve_image_url("/no/such/file.jpg") is None


# ----------------------------- encode_image ----------------------------

def test_encode_image_none():
    assert rag_system.encode_image(None) is None


def test_encode_image_missing_file():
    assert rag_system.encode_image("/no/such/file.png") is None


def test_encode_image_local_file(tmp_path):
    img = tmp_path / "pic.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\nfake-bytes")
    uri = rag_system.encode_image(str(img))
    assert uri is not None
    assert uri.startswith("data:image/png;base64,")


def test_encode_image_unknown_extension_defaults_to_jpeg(tmp_path):
    f = tmp_path / "blob.unknownext"
    f.write_bytes(b"1234")
    uri = rag_system.encode_image(str(f))
    assert uri.startswith("data:image/jpeg;base64,")


def test_encode_image_read_error_returns_none(tmp_path):
    # Un directorio "existe" pero no se puede leer como archivo -> None.
    d = tmp_path / "a_dir"
    d.mkdir()
    assert rag_system.encode_image(str(d)) is None


# --------------------------- generate_response -------------------------

def test_generate_response_no_docs():
    msg = rag_system.generate_response("q", [])
    assert msg == "No se encontraron productos relevantes para tu consulta."


def test_generate_response_text_success(monkeypatch):
    first = rag_system.FALLBACK_MODELS[0]
    client = _FakeClient({first: "respuesta del modelo"})
    monkeypatch.setattr(rag_system, "client", client)

    docs = [{"titulo": "Auriculares", "descripcion": "inalámbricos"}]
    out = rag_system.generate_response("¿cuál recomiendas?", docs)

    assert out == "respuesta del modelo"
    # El contexto enviado debe incluir el título del producto recuperado.
    _, messages = client.calls[0]
    prompt_text = messages[0]["content"][0]["text"]
    assert "Auriculares" in prompt_text
    assert "¿cuál recomiendas?" in prompt_text


def test_generate_response_uses_placeholder_for_nan_description(monkeypatch):
    first = rag_system.FALLBACK_MODELS[0]
    client = _FakeClient({first: "ok"})
    monkeypatch.setattr(rag_system, "client", client)

    docs = [{"titulo": "X", "descripcion": "nan"}]
    rag_system.generate_response("q", docs)
    prompt_text = client.calls[0][1][0]["content"][0]["text"]
    assert "Sin descripción" in prompt_text


def test_generate_response_includes_memory_block(monkeypatch):
    first = rag_system.FALLBACK_MODELS[0]
    client = _FakeClient({first: "ok"})
    monkeypatch.setattr(rag_system, "client", client)

    docs = [{"titulo": "X", "descripcion": "y"}]
    rag_system.generate_response("q", docs, memory="turno previo importante")
    prompt_text = client.calls[0][1][0]["content"][0]["text"]
    assert "turno previo importante" in prompt_text


def test_generate_response_falls_back_across_models(monkeypatch):
    models = rag_system.FALLBACK_MODELS
    client = _FakeClient(
        {models[0]: RuntimeError("429"), models[1]: "desde el respaldo"}
    )
    monkeypatch.setattr(rag_system, "client", client)

    out = rag_system.generate_response("q", [{"titulo": "T", "descripcion": "d"}])
    assert out == "desde el respaldo"
    assert [c[0] for c in client.calls] == [models[0], models[1]]


def test_generate_response_all_models_fail(monkeypatch):
    client = _FakeClient({m: RuntimeError("boom") for m in rag_system.FALLBACK_MODELS})
    monkeypatch.setattr(rag_system, "client", client)

    out = rag_system.generate_response("q", [{"titulo": "T", "descripcion": "d"}])
    assert out.startswith("Error al generar respuesta (todos los modelos gratuitos fallaron)")


def test_generate_response_image_query_adds_reference_image(monkeypatch, tmp_path):
    first = rag_system.FALLBACK_MODELS[0]
    client = _FakeClient({first: "ok"})
    monkeypatch.setattr(rag_system, "client", client)

    ref = tmp_path / "ref.png"
    ref.write_bytes(b"\x89PNGdata")
    docs = [{"titulo": "Similar", "descripcion": "d", "image_path": None}]

    rag_system.generate_response(str(ref), docs, query_type="image")

    content_blocks = client.calls[0][1][0]["content"]
    # El primer bloque debe ser la imagen de referencia del usuario.
    assert content_blocks[0]["type"] == "image_url"
    assert content_blocks[0]["image_url"]["url"].startswith("data:image/png;base64,")


def test_generate_response_attaches_product_images_up_to_max(monkeypatch):
    first = rag_system.FALLBACK_MODELS[0]
    client = _FakeClient({first: "ok"})
    monkeypatch.setattr(rag_system, "client", client)

    # 6 productos con imagen (URL http, pasa directo); max_images=2 recorta.
    docs = [
        {"titulo": f"P{i}", "descripcion": "d", "image_path": f"http://x/{i}.jpg"}
        for i in range(6)
    ]
    rag_system.generate_response("q", docs, max_images=2)

    content_blocks = client.calls[0][1][0]["content"]
    image_blocks = [b for b in content_blocks if b["type"] == "image_url"]
    assert len(image_blocks) == 2
    assert image_blocks[0]["image_url"]["url"] == "http://x/0.jpg"


# -------------------------------- ask ----------------------------------

def test_ask_returns_response_and_docs(monkeypatch):
    docs = [{"titulo": "Prod", "descripcion": "d", "image_path": None, "score_final": 0.9}]
    monkeypatch.setattr(rag_system, "search", lambda *a, **k: docs)

    first = rag_system.FALLBACK_MODELS[0]
    monkeypatch.setattr(rag_system, "client", _FakeClient({first: "respuesta"}))

    response, returned_docs = rag_system.ask("q", top_k=3)
    assert response == "respuesta"
    assert returned_docs == docs
