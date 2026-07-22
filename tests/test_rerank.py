"""Tests unitarios para rerank.py (re-ranking con Cross-Encoder)."""

import pytest

import rerank


class _FakeModel:
    """Modelo falso: devuelve scores fijos y recuerda los pares recibidos."""

    def __init__(self, scores):
        self._scores = scores
        self.received = None

    def predict(self, pairs):
        self.received = list(pairs)
        return self._scores


def test_empty_candidates_returns_same_object(monkeypatch):
    # No debe intentar cargar el modelo si no hay candidatos.
    def _boom():
        raise AssertionError("no debería cargar el modelo")

    monkeypatch.setattr(rerank, "_get_model", _boom)
    assert rerank.rerank("q", []) == []


def test_reorders_by_rerank_score_descending(monkeypatch):
    candidates = [
        {"titulo": "A", "descripcion": "desc a"},
        {"titulo": "B", "descripcion": "desc b"},
        {"titulo": "C", "descripcion": "desc c"},
    ]
    model = _FakeModel([0.1, 0.9, 0.5])
    monkeypatch.setattr(rerank, "_get_model", lambda: model)

    result = rerank.rerank("q", candidates)
    assert [c["titulo"] for c in result] == ["B", "C", "A"]
    assert [c["rerank_score"] for c in result] == [0.9, 0.5, 0.1]
    # El score debe quedar como float nativo.
    assert all(isinstance(c["rerank_score"], float) for c in result)


def test_top_k_truncates_result(monkeypatch):
    candidates = [
        {"titulo": "A", "descripcion": ""},
        {"titulo": "B", "descripcion": ""},
        {"titulo": "C", "descripcion": ""},
    ]
    model = _FakeModel([0.1, 0.9, 0.5])
    monkeypatch.setattr(rerank, "_get_model", lambda: model)

    result = rerank.rerank("q", candidates, top_k=2)
    assert [c["titulo"] for c in result] == ["B", "C"]


def test_builds_pairs_and_handles_nan_description(monkeypatch):
    candidates = [
        {"titulo": "Camera", "descripcion": "nan"},  # "nan" -> se ignora
        {"titulo": "Lens", "descripcion": "50mm prime"},
        {"titulo": "Tripod"},  # sin descripcion
    ]
    model = _FakeModel([0.3, 0.2, 0.1])
    monkeypatch.setattr(rerank, "_get_model", lambda: model)

    rerank.rerank("photo gear", candidates)

    assert model.received == [
        ("photo gear", "Camera."),
        ("photo gear", "Lens. 50mm prime"),
        ("photo gear", "Tripod."),
    ]


def test_truncates_document_text_to_512_chars(monkeypatch):
    long_desc = "x" * 1000
    candidates = [{"titulo": "T", "descripcion": long_desc}]
    model = _FakeModel([1.0])
    monkeypatch.setattr(rerank, "_get_model", lambda: model)

    rerank.rerank("q", candidates)

    _, doc_text = model.received[0]
    assert len(doc_text) == 512


def test_custom_keys(monkeypatch):
    candidates = [{"name": "Widget", "info": "shiny"}]
    model = _FakeModel([0.7])
    monkeypatch.setattr(rerank, "_get_model", lambda: model)

    result = rerank.rerank(
        "q", candidates, text_key="name", desc_key="info"
    )
    assert model.received[0] == ("q", "Widget. shiny")
    assert result[0]["rerank_score"] == 0.7
