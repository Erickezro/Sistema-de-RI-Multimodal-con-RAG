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

# ============================================================
# IMPORTACIÓN DE LA LIBRERÍA NECESARIA
# ============================================================
from sentence_transformers import CrossEncoder

# ============================================================
# CARGA DIFERIDA DEL MODELO (solo se carga la primera vez que se usa)
# ============================================================
_model = None

# Modelo preentrenado: MiniLM entrenado en MS MARCO, especializado en
# re-ranking de relevancia query-documento. Es rápido y eficaz para
# textos en inglés (el corpus de productos está en inglés).
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _get_model():
    """Carga el modelo CrossEncoder la primera vez que se llama,
    y devuelve la instancia. Esto evita recargarlo en cada llamada."""
    global _model
    if _model is None:
        _model = CrossEncoder(MODEL_NAME)
    return _model


# ============================================================
# FUNCIÓN PRINCIPAL DE RE-RANKING
# ============================================================
def rerank(query, candidates, top_k=None, text_key="titulo", desc_key="descripcion"):
    """
    Reordena los candidatos (resultados de la búsqueda vectorial)
    usando un Cross-Encoder, que evalúa la relevancia de cada par
    (query, documento) de forma conjunta y directa.

    Parámetros:
        query: texto de la consulta del usuario.
        candidates: lista de diccionarios (resultados de search()).
        top_k: si se da, recorta la lista final a los primeros top_k.
        text_key, desc_key: claves dentro de cada candidato donde
            encontrar el título y la descripción del producto.

    Devuelve:
        La misma lista de candidatos, pero reordenada por el score del
        cross-encoder, y cada elemento tendrá un campo 'rerank_score'
        con la puntuación asignada.
    """
    if not candidates:
        return candidates

    # Cargamos el modelo (si no lo estaba ya)
    model = _get_model()

    # Construimos los pares (query, texto_documento) para el cross-encoder
    pairs = []
    for c in candidates:
        titulo = c.get(text_key, "") or ""
        desc = c.get(desc_key, "") or ""
        # Si la descripción es 'nan', la tratamos como vacía
        if str(desc).lower() == "nan":
            desc = ""
        # Limitamos la longitud del documento a 512 caracteres para no
        # exceder el límite del modelo y mantener velocidad.
        doc_text = f"{titulo}. {desc}".strip()[:512]
        pairs.append((query, doc_text))

    # El modelo predice scores de relevancia para todos los pares
    scores = model.predict(pairs)

    # Añadimos el score a cada candidato
    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)

    # Ordenamos de mayor a menor puntuación
    candidates = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)

    # Si se pidió un top_k, recortamos la lista
    if top_k:
        candidates = candidates[:top_k]

    return candidates