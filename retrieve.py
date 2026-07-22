"""
Módulo de recuperación (retrieve.py)

Función principal: search() - realiza búsquedas multimodales (texto o imagen)
contra las colecciones vectoriales de CLIP. Soporta:
- Búsqueda por texto o imagen (usando el embedding correspondiente)
- Query Expansion (genera variantes de la consulta con LLM y combina resultados)
- Re-ranking con Cross-Encoder (refina el orden de los candidatos)
- Relevance Feedback (aplica boost/penalización según likes/dislikes previos)

Devuelve una lista de productos enriquecida con scores y metadatos.
"""

# ============================================================
# IMPORTACIONES Y CARGA DE DATOS
# ============================================================
import pandas as pd
from generate_embeddings import embed_text, embed_image
from vector_db import text_collection, image_collection
from feedback_store import load_feedback, get_boost
from rerank import rerank as rerank_candidates
from query_expansion import expand_query

# Cargamos el catálogo de productos (indexado por product_id) para
# poder acceder a títulos, descripciones y rutas de imagen.
products = pd.read_csv("data/processed/products.csv").set_index("product_id")


# ============================================================
# FUNCIÓN AUXILIAR: BÚSQUEDA VECTORIAL EN AMBAS COLECCIONES
# ============================================================
def _query_collections(query_vec, n_fetch):
    """
    Dado un vector de consulta (texto o imagen, ambos en el mismo espacio
    CLIP), consulta las dos colecciones vectoriales (texto e imagen) y
    devuelve dos diccionarios con la similitud (coseno, convertida de distancia)
    para cada producto en cada colección.
    """
    res_text = text_collection.query(query_embeddings=[query_vec], n_results=n_fetch)
    res_image = image_collection.query(query_embeddings=[query_vec], n_results=n_fetch)
    # Chroma devuelve distancias (1 - similitud coseno), así que recuperamos similitud.
    sims_text = {pid: 1 - d for pid, d in zip(res_text["ids"][0], res_text["distances"][0])}
    sims_image = {pid: 1 - d for pid, d in zip(res_image["ids"][0], res_image["distances"][0])}
    return sims_text, sims_image


# ============================================================
# FUNCIÓN PRINCIPAL: SEARCH
# ============================================================
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
    # --- 1. Preparación de variantes (con o sin expansión) ---
    variants = [query]  # La consulta original siempre se incluye
    if query_type == "text" and use_expansion:
        variants += expand_query(query, n_variants=n_expansions)  # Puede devolver [] si falla

    # --- 2. Búsqueda vectorial para todas las variantes ---
    sims_text, sims_image = {}, {}
    n_fetch = max(top_k * 4, 20)  # Traemos más de lo necesario para luego re-rankear o combinar

    for variant in variants:
        # Generamos el embedding correspondiente (texto o imagen)
        if query_type == "text":
            vec = embed_text(variant).tolist()
        else:
            vec = embed_image(variant).tolist()

        # Consultamos ambas colecciones con este vector
        v_sims_text, v_sims_image = _query_collections(vec, n_fetch)

        # Combinamos resultados: para cada producto, nos quedamos con la MAYOR similitud
        # obtenida entre todas las variantes (esto mejora el recall).
        for pid, s in v_sims_text.items():
            sims_text[pid] = max(sims_text.get(pid, 0.0), s)
        for pid, s in v_sims_image.items():
            sims_image[pid] = max(sims_image.get(pid, 0.0), s)

    # --- 3. Cálculo de scores finales (combinación + feedback) ---
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

    # Ordenamos por score final descendente
    scored.sort(key=lambda x: x[1], reverse=True)

    # --- 4. Preparación del pool para re-ranking (si aplica) ---
    apply_rerank = use_rerank and query_type == "text"
    pool_n = max(rerank_pool_size, top_k) if apply_rerank else top_k
    scored = scored[:pool_n]   # Recortamos al tamaño del pool (o top_k si no hay rerank)

    # --- 5. Construcción de la lista de resultados (metadatos) ---
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

    # --- 6. Re-ranking con Cross-Encoder (solo texto) ---
    if apply_rerank:
        results = rerank_candidates(query, results, top_k=top_k)
    else:
        results = results[:top_k]

    return results


# ============================================================
# BLOQUE DE PRUEBAS (ejecutar directamente con python retrieve.py)
# ============================================================
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

    # Prueba (comentada) para búsqueda por imagen:
    # for r in search("data/raw/images/B00FTCMR96.jpg", query_type="image", top_k=5):
    #     print(f"[{r['score_final']}] {r['titulo']} (texto={r['score_texto']}, imagen={r['score_imagen']})")