"""Tests unitarios para evaluate.py (métricas de recuperación)."""

import math

import numpy as np
import pandas as pd
import pytest

import evaluate


# ------------------------- precision_at_k -------------------------

def test_precision_at_k_zero_k():
    assert evaluate.precision_at_k(["a", "b"], ["a"], 0) == 0.0


def test_precision_at_k_partial_match():
    # 1 de los primeros 2 recuperados es relevante -> 0.5
    assert evaluate.precision_at_k(["a", "x"], ["a", "b"], 2) == 0.5


def test_precision_at_k_no_overlap():
    assert evaluate.precision_at_k(["x", "y"], ["a", "b"], 2) == 0.0


def test_precision_at_k_all_relevant():
    assert evaluate.precision_at_k(["a", "b"], ["a", "b"], 2) == 1.0


def test_precision_at_k_ignores_beyond_k():
    # El match está en la posición 3, fuera del top-2.
    assert evaluate.precision_at_k(["x", "y", "a"], ["a"], 2) == 0.0


# --------------------------- recall_at_k --------------------------

def test_recall_at_k_empty_relevant():
    assert evaluate.recall_at_k(["a"], [], 1) == 0.0


def test_recall_at_k_partial():
    # 1 de 2 relevantes recuperado -> 0.5
    assert evaluate.recall_at_k(["a", "x"], ["a", "b"], 2) == 0.5


def test_recall_at_k_full():
    assert evaluate.recall_at_k(["a", "b"], ["a", "b"], 2) == 1.0


def test_recall_at_k_cutoff_limits_retrieved():
    # Solo miramos el primer recuperado; 'b' queda fuera del corte.
    assert evaluate.recall_at_k(["a", "b"], ["a", "b"], 1) == 0.5


# ---------------------------- ndcg_at_k ---------------------------

def test_ndcg_at_k_empty_retrieved():
    assert evaluate.ndcg_at_k([], {"a": 3}, 5) == 0.0


def test_ndcg_at_k_no_relevance_returns_zero():
    # Ningún doc recuperado tiene relevancia -> ideal e ideal-dcg = 0.
    assert evaluate.ndcg_at_k(["x", "y"], {"x": 0, "y": 0}, 2) == 0.0


def test_ndcg_at_k_perfect_ranking_is_one():
    scores = {"a": 3, "b": 2, "c": 1}
    assert evaluate.ndcg_at_k(["a", "b", "c"], scores, 3) == pytest.approx(1.0)


def test_ndcg_at_k_suboptimal_ranking_less_than_one():
    scores = {"a": 3, "b": 2}
    # Orden invertido: relevancia alta al final -> DCG < IDCG.
    val = evaluate.ndcg_at_k(["b", "a"], scores, 2)
    assert 0.0 < val < 1.0

    # Verificamos el valor exacto contra la fórmula.
    dcg = 2 / math.log2(2) + 3 / math.log2(3)
    idcg = 3 / math.log2(2) + 2 / math.log2(3)
    assert val == pytest.approx(dcg / idcg)


# ----------------------------- evaluate ---------------------------

def test_evaluate_end_to_end(monkeypatch):
    """evaluate() debe agregar las métricas por k usando search()."""
    qrels = pd.DataFrame(
        {
            "query": ["q1", "q1", "q2"],
            "product_id": ["a", "b", "c"],
            "relevance": [3, 2, 3],
        }
    )

    # search() se sustituye: q1 recupera [a, b] (perfecto), q2 recupera
    # [z, c] (el relevante 'c' aparece en segunda posición).
    def fake_search(query, query_type="text", top_k=5):
        mapping = {
            "q1": ["a", "b", "x", "y"],
            "q2": ["z", "c", "y", "x"],
        }
        return [{"product_id": pid} for pid in mapping[query][:top_k]]

    monkeypatch.setattr(evaluate, "search", fake_search)

    averages = evaluate.evaluate(qrels, k_values=[1, 2])

    assert set(averages.keys()) == {1, 2}
    for k in (1, 2):
        assert set(averages[k].keys()) == {"precision", "recall", "ndcg"}

    # precision@1: q1 -> 'a' relevante (1.0); q2 -> 'z' no relevante (0.0) => 0.5
    assert averages[1]["precision"] == pytest.approx(0.5)
    # precision@2: q1 -> 2/2 = 1.0; q2 -> 1/2 = 0.5 => 0.75
    assert averages[2]["precision"] == pytest.approx(0.75)

    # recall@2: q1 recupera ambos relevantes (1.0); q2 recupera su único
    # relevante (1.0) => 1.0
    assert averages[2]["recall"] == pytest.approx(1.0)

    # Todas las métricas deben ser numéricas.
    assert isinstance(averages[1]["ndcg"], (float, np.floating))
