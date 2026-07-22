"""Tests unitarios para query_expansion.py (expansión de consultas)."""

from types import SimpleNamespace

import pytest

import query_expansion


def _make_response(content):
    """Construye un objeto con la forma de la respuesta de OpenAI."""
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


class _FakeClient:
    """Cliente falso: devuelve/lanza según lo configurado por modelo."""

    def __init__(self, behaviors):
        # behaviors: dict model_name -> str (contenido) o Exception a lanzar.
        self._behaviors = behaviors
        self.calls = []
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    def _create(self, model, messages, temperature, max_tokens):
        self.calls.append(model)
        behavior = self._behaviors[model]
        if isinstance(behavior, Exception):
            raise behavior
        return _make_response(behavior)


def test_empty_query_returns_empty(monkeypatch):
    monkeypatch.setattr(query_expansion, "_client", _FakeClient({}))
    assert query_expansion.expand_query("") == []


def test_whitespace_query_returns_empty(monkeypatch):
    monkeypatch.setattr(query_expansion, "_client", _FakeClient({}))
    assert query_expansion.expand_query("   ") == []


def test_no_client_returns_empty(monkeypatch):
    monkeypatch.setattr(query_expansion, "_client", None)
    assert query_expansion.expand_query("headphones") == []


def test_successful_expansion_parses_and_cleans(monkeypatch):
    first = query_expansion.EXPANSION_MODELS[0]
    content = "- wireless earbuds\n2. bluetooth headset\n* over-ear headphones"
    monkeypatch.setattr(
        query_expansion, "_client", _FakeClient({first: content})
    )
    variants = query_expansion.expand_query("headphones", n_variants=3)
    assert variants == [
        "wireless earbuds",
        "bluetooth headset",
        "over-ear headphones",
    ]


def test_expansion_filters_out_original_query(monkeypatch):
    first = query_expansion.EXPANSION_MODELS[0]
    content = "headphones\nwireless earbuds"  # la 1ra es igual a la original
    monkeypatch.setattr(
        query_expansion, "_client", _FakeClient({first: content})
    )
    variants = query_expansion.expand_query("Headphones")
    assert variants == ["wireless earbuds"]


def test_expansion_respects_n_variants_limit(monkeypatch):
    first = query_expansion.EXPANSION_MODELS[0]
    content = "one\ntwo\nthree\nfour\nfive"
    monkeypatch.setattr(
        query_expansion, "_client", _FakeClient({first: content})
    )
    variants = query_expansion.expand_query("q", n_variants=2)
    assert variants == ["one", "two"]


def test_falls_back_to_next_model_on_error(monkeypatch):
    models = query_expansion.EXPANSION_MODELS
    behaviors = {
        models[0]: RuntimeError("429 rate limit"),
        models[1]: "alt query",
    }
    client = _FakeClient(behaviors)
    monkeypatch.setattr(query_expansion, "_client", client)
    variants = query_expansion.expand_query("q")
    assert variants == ["alt query"]
    # Debe haber intentado el primer modelo antes de pasar al segundo.
    assert client.calls == [models[0], models[1]]


def test_returns_empty_when_all_models_fail(monkeypatch):
    behaviors = {m: RuntimeError("boom") for m in query_expansion.EXPANSION_MODELS}
    monkeypatch.setattr(query_expansion, "_client", _FakeClient(behaviors))
    assert query_expansion.expand_query("q") == []


def test_empty_content_skips_to_next_model(monkeypatch):
    models = query_expansion.EXPANSION_MODELS
    behaviors = {
        models[0]: "",  # respuesta vacía -> no genera variantes
        models[1]: "good variant",
    }
    monkeypatch.setattr(query_expansion, "_client", _FakeClient(behaviors))
    assert query_expansion.expand_query("q") == ["good variant"]
