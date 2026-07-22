"""
Re-ranking con Cross-Encoder

La búsqueda vectorial (CLIP) es un "bi-encoder": el embedding de la query
y el de cada producto se calculan por separado y se comparan con
similitud coseno. Es rápido y escala a todo el corpus, pero pierde
precisión porque nunca "ve" la query y el documento juntos.

Un Cross-Encoder sí procesa (query, documento) como un solo input y
devuelve un score de relevancia directo -- mucho más preciso, pero
demasiado lento para correr sobre todo el corpus. Por eso el patrón
estándar es: usar el bi-encoder para traer un conjunto amplio de
candidatos (ej. 30), y el cross-encoder solo para reordenar esos
candidatos antes de quedarnos con el Top-k final.

Solo aplica a queries de TEXTO -- el cross-encoder usado es de
texto-texto, no compara imágenes.
"""

from sentence_transformers import CrossEncoder

_model = None

# MiniLM entrenado en MS MARCO: rápido y con buen desempeño para
# re-ranking de relevancia query-documento en inglés (el corpus de
# productos está en inglés).
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _get_model():
    global _model
    if _model is None:
        _model = CrossEncoder(MODEL_NAME)
    return _model


def rerank(query, candidates, top_k=None, text_key="titulo", desc_key="descripcion"):
    """
    query: string de texto de la consulta del usuario.
    candidates: lista de dicts (resultados de search()), cada uno debe
        tener al menos `text_key` (título) y opcionalmente `desc_key`.
    top_k: si se da, recorta el resultado final a los primeros top_k
        después de reordenar.

    Devuelve la misma lista reordenada por relevancia real (no por
    similitud coseno), agregando el campo 'rerank_score' a cada item.
    """
    if not candidates:
        return candidates

    model = _get_model()

    pairs = []
    for c in candidates:
        titulo = c.get(text_key, "") or ""
        desc = c.get(desc_key, "") or ""
        if str(desc).lower() == "nan":
            desc = ""
        doc_text = f"{titulo}. {desc}".strip()[:512]  # limitar longitud del input
        pairs.append((query, doc_text))

    scores = model.predict(pairs)

    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)

    candidates = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)

    if top_k:
        candidates = candidates[:top_k]
    return candidates