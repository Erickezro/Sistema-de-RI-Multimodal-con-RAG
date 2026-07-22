"""Tests unitarios para feedback_store.py (Relevance Feedback)."""

import json

import pytest

import feedback_store


@pytest.fixture
def feedback_path(tmp_path, monkeypatch):
    """Redirige la persistencia a un JSON temporal aislado por test."""
    path = tmp_path / "sub" / "feedback.json"
    monkeypatch.setattr(feedback_store, "FEEDBACK_PATH", str(path))
    return path


def test_load_feedback_missing_file_returns_empty(feedback_path):
    assert not feedback_path.exists()
    assert feedback_store.load_feedback() == {}


def test_save_and_load_roundtrip(feedback_path):
    data = {"p1": {"likes": 2, "dislikes": 1}, "ñ": {"likes": 0, "dislikes": 0}}
    feedback_store.save_feedback(data)

    # El archivo (y su carpeta padre) se crea y conserva unicode legible.
    assert feedback_path.exists()
    raw = feedback_path.read_text(encoding="utf-8")
    assert "ñ" in raw
    assert json.loads(raw) == data

    assert feedback_store.load_feedback() == data


def test_add_feedback_creates_new_entry_like(feedback_path):
    feedback_store.add_feedback("prod-a", liked=True)
    data = feedback_store.load_feedback()
    assert data["prod-a"] == {"likes": 1, "dislikes": 0}


def test_add_feedback_creates_new_entry_dislike(feedback_path):
    feedback_store.add_feedback("prod-a", liked=False)
    data = feedback_store.load_feedback()
    assert data["prod-a"] == {"likes": 0, "dislikes": 1}


def test_add_feedback_increments_existing_entry(feedback_path):
    feedback_store.add_feedback("prod-a", liked=True)
    feedback_store.add_feedback("prod-a", liked=True)
    feedback_store.add_feedback("prod-a", liked=False)
    data = feedback_store.load_feedback()
    assert data["prod-a"] == {"likes": 2, "dislikes": 1}


def test_get_boost_unknown_product_is_zero(feedback_path):
    assert feedback_store.get_boost("desconocido") == 0.0


def test_get_boost_positive_net(feedback_path):
    data = {"p": {"likes": 3, "dislikes": 0}}  # net = 3
    boost = feedback_store.get_boost("p", data)
    assert boost == pytest.approx(feedback_store.ALPHA * 3)


def test_get_boost_negative_net(feedback_path):
    data = {"p": {"likes": 1, "dislikes": 3}}  # net = -2
    boost = feedback_store.get_boost("p", data)
    assert boost == pytest.approx(feedback_store.ALPHA * -2)


def test_get_boost_clamped_to_positive_cap(feedback_path):
    data = {"p": {"likes": 1000, "dislikes": 0}}
    assert feedback_store.get_boost("p", data) == pytest.approx(feedback_store.CAP)


def test_get_boost_clamped_to_negative_cap(feedback_path):
    data = {"p": {"likes": 0, "dislikes": 1000}}
    assert feedback_store.get_boost("p", data) == pytest.approx(-feedback_store.CAP)


def test_get_boost_loads_from_disk_when_data_is_none(feedback_path):
    feedback_store.save_feedback({"p": {"likes": 5, "dislikes": 0}})
    # Sin pasar `data`, debe leer el JSON persistido.
    assert feedback_store.get_boost("p") == pytest.approx(feedback_store.ALPHA * 5)


def test_get_stats_default(feedback_path):
    assert feedback_store.get_stats("nuevo") == {"likes": 0, "dislikes": 0}


def test_get_stats_existing(feedback_path):
    feedback_store.save_feedback({"p": {"likes": 4, "dislikes": 2}})
    assert feedback_store.get_stats("p") == {"likes": 4, "dislikes": 2}
