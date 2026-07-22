# ------------------------------------------------------------------
# Función de busqueda: acepta una query de TEXTO o de IMAGEN
# ------------------------------------------------------------------
from generate_embeddings import embed_text, embed_image
from vector_db import text_collection, image_collection
from feedback_store import load_feedback, get_boost
from rerank import rerank as rerank_candidates
from query_expansion import expand_query
from data_utils import load_products

products = load_products(index_by_id=True)


def _query_collections(query_vec, n_fetch):
    """Corre la búsqueda vectorial contra ambas colecciones para un vector dado."""
    res_text = text_collection.query(query_embeddings=[query_vec], n_results=n_fetch)
    res_image = image_collection.query(query_embeddings=[query_vec], n_results=n_fetch)
    sims_text = {pid: 1 - d for pid, d in zip(res_text["ids"][0], res_text["distances"][0])}
    sims_image = {pid: 1 - d for pid, d in zip(res_image["ids"][0], res_image["distances"][0])}
    return sims_text, sims_image


def search(query, query_type="text", top_k=5, text_weight=0.5, image_weight=0.5,
           use_feedback=True, use_rerank=True, rerank_pool_size=30,
           use_expansion=False, n_expansions=3):
    """
    query_type = "text"--> query es un string (ej. "wireless headphones")
    query_type = "image"--> query es una ruta a una imagen (ej. "foto_usuario.jpg")

    En ambos casos se busca contra LAS DOS colecciones (texto e imagen),
    porque CLIP pone texto e imagen en el mismo espacio vectorial.

    use_feedback: si True, aplica un pequeño boost/penalización a productos
    que hayan recibido relevance feedback (👍/👎) en consultas anteriores.

    use_rerank: si True (y query_type == "text"), refina el ranking inicial
    de la búsqueda vectorial usando un Cross-Encoder sobre un pool más
    amplio de candidatos (rerank_pool_size) antes de devolver el Top-k final.

    use_expansion: si True (y query_type == "text"), genera hasta
    n_expansions reformulaciones de la query con un LLM y busca también
    con esas variantes, combinando resultados por el score MÁXIMO obtenido
    entre todas las formulaciones para cada producto (mejora el recall
    cuando el usuario usa vocabulario distinto al del catálogo). Si la
    generación de variantes falla (ej. rate limit del LLM), se degrada
    de forma segura a buscar solo con la query original -- search() nunca
    lanza una excepción por esto.

    Devuelve solo la lista de resultados (para no romper la firma que ya
    usan rag_system.ask() y evaluate.py). Las variantes usadas quedan
    disponibles llamando aparte a query_expansion.expand_query(query) si
    se necesitan mostrar en la interfaz.
    """
    variants = [query]
    if query_type == "text" and use_expansion:
        variants += expand_query(query, n_variants=n_expansions)

    sims_text, sims_image = {}, {}
    n_fetch = max(top_k * 4, 20)

    for variant in variants:
        if query_type == "text":
            vec = embed_text(variant).tolist()
        else:
            vec = embed_image(variant).tolist()

        v_sims_text, v_sims_image = _query_collections(vec, n_fetch)

        for pid, s in v_sims_text.items():
            sims_text[pid] = max(sims_text.get(pid, 0.0), s)
        for pid, s in v_sims_image.items():
            sims_image[pid] = max(sims_image.get(pid, 0.0), s)

    all_ids = set(sims_text) | set(sims_image)

    feedback_data = load_feedback() if use_feedback else {}

    scored = []
    for pid in all_ids:
        s_text = sims_text.get(pid, 0.0)
        s_image = sims_image.get(pid, 0.0)
        base_score = text_weight * s_text + image_weight * s_image
        boost = get_boost(pid, feedback_data) if use_feedback else 0.0
        final_score = base_score + boost
        scored.append((pid, final_score, s_text, s_image, boost))

    scored.sort(key=lambda x: x[1], reverse=True)

    # Si vamos a re-rankear, traemos un pool más amplio de candidatos
    # (el bi-encoder es una recuperación inicial de "grueso", el
    # cross-encoder refina); si no, nos quedamos solo con el top_k directo.
    apply_rerank = use_rerank and query_type == "text"
    pool_n = max(rerank_pool_size, top_k) if apply_rerank else top_k
    scored = scored[:pool_n]

    results = []
    for pid, score, s_text, s_image, boost in scored:
        row = products.loc[pid]
        results.append({
            "product_id": pid,
            "score_final": round(score, 4),
            "score_texto": round(s_text, 4),
            "score_imagen": round(s_image, 4),
            "feedback_boost": round(boost, 4),
            "titulo": row["product_title"],
            "descripcion": row.get("product_description", ""),
            "image_path": row["image_path"],
        })

    if apply_rerank:
        results = rerank_candidates(query, results, top_k=top_k)
    else:
        results = results[:top_k]

    return results


if __name__ == "__main__":
    # Prueba rápida: búsqueda por texto (con re-ranking)
    print("\n--- Búsqueda por texto: 'wireless headphones' (con re-ranking) ---")
    for r in search("wireless headphones", query_type="text", top_k=5, use_rerank=True):
        print(f"[vec={r['score_final']} | rerank={r.get('rerank_score', 'N/A')}] {r['titulo']}")

    # Prueba rápida: la misma búsqueda SIN re-ranking, para comparar el orden
    print("\n--- Misma búsqueda SIN re-ranking (solo bi-encoder CLIP) ---")
    for r in search("wireless headphones", query_type="text", top_k=5, use_rerank=False):
        print(f"[vec={r['score_final']}] {r['titulo']}")

    # Prueba rápida: Query Expansion -- comparar CON y SIN expansión
    print("\n--- Query Expansion: variantes generadas para 'earbuds' ---")
    variants = expand_query("earbuds", n_variants=3)
    print(f"Variantes: {variants}")

    print("\n--- Búsqueda 'earbuds' SIN expansión ---")
    for r in search("earbuds", query_type="text", top_k=5, use_expansion=False):
        print(f"[vec={r['score_final']}] {r['titulo']}")

    print("\n--- Búsqueda 'earbuds' CON expansión ---")
    for r in search("earbuds", query_type="text", top_k=5, use_expansion=True):
        print(f"[vec={r['score_final']}] {r['titulo']}")

    # Prueba rápida: búsqueda por imagen (re-ranking y expansion se ignoran automáticamente)
    # for r in search("data/raw/images/B00FTCMR96.jpg", query_type="image", top_k=5):
    #     print(f"[{r['score_final']}] {r['titulo']} (texto={r['score_texto']}, imagen={r['score_imagen']})")