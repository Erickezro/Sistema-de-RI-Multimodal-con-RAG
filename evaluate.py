"""
Evaluación con Precision@k, Recall@k, NDCG@k
 
Nota: la evaluación mide la calidad de la RECUPERACIÓN (search),
no de la generación con el LLM. Por eso usamos search() directamente
en vez de ask() -- evita llamadas innecesarias a la API del LLM
(que además son lentas y pueden colgarse por rate limits).
"""
 
import pandas as pd
import numpy as np
from tqdm import tqdm
from retrieve import search
 

# ============================================================
# FUNCIONES DE MÉTRICAS DE EVALUACIÓN
# ============================================================

def precision_at_k(retrieved, relevant, k):
    """Proporción de documentos relevantes entre los primeros k recuperados.
       retrieved: lista de IDs recuperados (en orden)
       relevant: conjunto de IDs que son realmente relevantes para la query
       k: número de resultados a considerar
    """
    if k == 0:
        return 0.0
    retrieved_set = set(retrieved[:k])
    relevant_set = set(relevant)
    return len(retrieved_set & relevant_set) / k
 
 
def recall_at_k(retrieved, relevant, k):
    """Proporción de todos los documentos relevantes que han sido recuperados en los primeros k.
       Es la capacidad del sistema para encontrar todos los relevantes.
    """
    retrieved_set = set(retrieved[:k])
    relevant_set = set(relevant)
    if len(relevant_set) == 0:
        return 0.0
    return len(retrieved_set & relevant_set) / len(relevant_set)
 
 
def ndcg_at_k(retrieved, relevant_scores, k):
    """Normalized Discounted Cumulative Gain: mide la calidad del ranking,
       dando más peso a los documentos relevantes que aparecen en posiciones altas.
       relevant_scores: diccionario {doc_id: puntuación de relevancia (cuanto mayor, mejor)}
    """
    # Puntuaciones de los primeros k documentos recuperados (0 si no están en el diccionario)
    rel_scores = [relevant_scores.get(doc, 0) for doc in retrieved[:k]]
    if not rel_scores:
        return 0.0
    # Orden ideal: los mejores puntajes posibles en orden descendente
    ideal = sorted(relevant_scores.values(), reverse=True)[:k]
    if sum(ideal) == 0:
        return 0.0
    # DCG: suma de (relevancia / log2(posición+1)) (posición empieza en 0 -> i+2)
    dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(rel_scores))
    # IDCG: DCG del ranking ideal
    idcg = sum(ideal[i] / np.log2(i + 2) for i in range(len(ideal)))
    return dcg / idcg if idcg > 0 else 0.0
 
 
# ============================================================
# FUNCIÓN PRINCIPAL DE EVALUACIÓN
# ============================================================

def evaluate(qrels, k_values=[1, 3, 5, 10]):
    """
    Evalúa el sistema de recuperación (retrieve.search) para cada query en qrels.
    qrels: DataFrame con columnas 'query', 'product_id', 'relevance' (puntuación numérica)
    k_values: lista de valores de k para calcular las métricas.
    Retorna: diccionario con promedios de precisión, recall y NDCG para cada k.
    """
    # Estructura para acumular resultados
    results = {k: {"precision": [], "recall": [], "ndcg": []} for k in k_values}
 
    unique_queries = qrels["query"].unique()
 
    # Iteramos sobre cada query única
    for query in tqdm(unique_queries, desc="Evaluando queries"):
        # Obtenemos las filas de qrels correspondientes a esta query
        subset = qrels[qrels["query"] == query]
        # Conjunto de productos relevantes (IDs)
        relevant_products = set(subset["product_id"].tolist())
        # Diccionario de puntuaciones de relevancia (para NDCG)
        # CUIDADO: antes había un bug al alinear, ahora usamos el mismo subset
        relevant_scores = dict(zip(subset["product_id"], subset["relevance"]))
 
        # Llamamos al sistema de búsqueda (sin generación de texto) con top_k = max(k_values)
        docs = search(query, query_type="text", top_k=max(k_values))
        # Extraemos solo los IDs de los documentos recuperados
        retrieved_ids = [doc["product_id"] for doc in docs]
 
        # Para cada k, calculamos las métricas y las acumulamos en results
        for k in k_values:
            results[k]["precision"].append(precision_at_k(retrieved_ids, relevant_products, k))
            results[k]["recall"].append(recall_at_k(retrieved_ids, relevant_products, k))
            results[k]["ndcg"].append(ndcg_at_k(retrieved_ids, relevant_scores, k))
 
    # Calculamos el promedio de cada métrica para cada k
    averages = {k: {m: np.mean(results[k][m]) for m in ["precision", "recall", "ndcg"]} for k in k_values}
    return averages
 
 
# ============================================================
# EJECUCIÓN PRINCIPAL (cuando se corre el script directamente)
# ============================================================

if __name__ == "__main__":
    # Cargamos el archivo qrels.csv (generado por clean_corpus.py)
    qrels = pd.read_csv("data/processed/qrels.csv")
    # Mapeamos las etiquetas ESCI a puntuaciones numéricas para NDCG
    # E = Exact (más relevante), S = Substitute, C = Complement, I = Irrelevant
    label_map = {"E": 3, "S": 2, "C": 1, "I": 0}
    qrels["relevance"] = qrels["esci_label"].map(label_map)
 
    print(f"Total filas en qrels: {len(qrels)}")
    print(f"Queries únicas a evaluar: {qrels['query'].nunique()}")
 
    k_values = [1, 3, 5, 10]
    avg = evaluate(qrels, k_values)
 
    # Mostramos los resultados en tabla
    print("\n=== RESULTADOS DE EVALUACIÓN ===")
    print("k\tPrecision@k\tRecall@k\tNDCG@k")
    for k in k_values:
        print(f"{k}\t{avg[k]['precision']:.4f}\t\t{avg[k]['recall']:.4f}\t\t{avg[k]['ndcg']:.4f}")