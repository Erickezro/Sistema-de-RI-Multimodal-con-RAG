# ------------------------------------------------------------------
# Función de busqueda: acepta una query de TEXTO o de IMAGEN
# ------------------------------------------------------------------
import pandas as pd

from generate_embeddings import embed_text, embed_image
from vector_db import text_collection, image_collection
from feedback_store import load_feedback, get_boost
from rerank import rerank as rerank_candidates

products = pd.read_csv("data/processed/products.csv").set_index("product_id")


def search(query, query_type="text", top_k=5, text_weight=0.5, image_weight=0.5, use_feedback=True, use_rerank=True, rerank_pool_size=30):
    """
    query_type = "text"--> query es un string (ej. "wireless headphones")
    query_type = "image"--> query es una ruta a una imagen (ej. "foto_usuario.jpg")

    En ambos casos se busca contra LAS DOS colecciones (texto e imagen),
    porque CLIP pone texto e imagen en el mismo espacio vectorial.

    use_feedback: si True, aplica un pequeño boost/penalización a productos
    que hayan recibido relevance feedback (👍/👎) en consultas anteriores.

    use_rerank: si True (y query_type == "text"), refina el ranking inicial
    de la búsqueda vectorial usando un Cross-Encoder sobre un pool más
    amplio de candidatos (rerank_pool_size) antes de devolver el Top-k
    final. Solo aplica a queries de texto -- el cross-encoder compara
    texto contra texto, no imágenes.
    """
    if query_type == "text":
        query_vec = embed_text(query).tolist()
    elif query_type == "image":
        query_vec = embed_image(query).tolist()
    else:
        raise ValueError("query_type debe ser 'text' o 'image'")

    n_fetch = max(top_k * 4, 20)
    res_text = text_collection.query(query_embeddings=[query_vec], n_results=n_fetch)
    res_image = image_collection.query(query_embeddings=[query_vec], n_results=n_fetch)

    # distance -> similitud coseno (porque space="cosine")
    sims_text = {pid: 1 - d for pid, d in zip(res_text["ids"][0], res_text["distances"][0])}
    sims_image = {pid: 1 - d for pid, d in zip(res_image["ids"][0], res_image["distances"][0])}

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
        
    # Prueba rápida: búsqueda por imagen
    print("\n--- Búsqueda por imagen ---")
    for r in search("data/raw/images/B00FTCMR96.jpg", query_type="image", top_k=5):
        print(f"[{r['score_final']}] {r['titulo']} (texto={r['score_texto']}, imagen={r['score_imagen']})")