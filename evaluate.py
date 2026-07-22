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
from data_utils import QRELS_CSV
 
 
def precision_at_k(retrieved, relevant, k):
    if k == 0:
        return 0.0
    retrieved_set = set(retrieved[:k])
    relevant_set = set(relevant)
    return len(retrieved_set & relevant_set) / k
 
 
def recall_at_k(retrieved, relevant, k):
    retrieved_set = set(retrieved[:k])
    relevant_set = set(relevant)
    if len(relevant_set) == 0:
        return 0.0
    return len(retrieved_set & relevant_set) / len(relevant_set)
 
 
def ndcg_at_k(retrieved, relevant_scores, k):
    rel_scores = [relevant_scores.get(doc, 0) for doc in retrieved[:k]]
    if not rel_scores:
        return 0.0
    ideal = sorted(relevant_scores.values(), reverse=True)[:k]
    if sum(ideal) == 0:
        return 0.0
    dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(rel_scores))
    idcg = sum(ideal[i] / np.log2(i + 2) for i in range(len(ideal)))
    return dcg / idcg if idcg > 0 else 0.0
 
 
def evaluate(qrels, k_values=[1, 3, 5, 10]):
    results = {k: {"precision": [], "recall": [], "ndcg": []} for k in k_values}
 
    unique_queries = qrels["query"].unique()
 
    for query in tqdm(unique_queries, desc="Evaluando queries"):
        subset = qrels[qrels["query"] == query]
        relevant_products = set(subset["product_id"].tolist())
        # Antes: el zip mezclaba el product_id filtrado con la columna
        # 'relevance' SIN filtrar -> quedaban desalineados. Se corrige
        # usando el mismo 'subset' para ambos lados del zip.
        relevant_scores = dict(zip(subset["product_id"], subset["relevance"]))
 
        docs = search(query, query_type="text", top_k=max(k_values))
        retrieved_ids = [doc["product_id"] for doc in docs]
 
        for k in k_values:
            results[k]["precision"].append(precision_at_k(retrieved_ids, relevant_products, k))
            results[k]["recall"].append(recall_at_k(retrieved_ids, relevant_products, k))
            results[k]["ndcg"].append(ndcg_at_k(retrieved_ids, relevant_scores, k))
 
    averages = {k: {m: np.mean(results[k][m]) for m in ["precision", "recall", "ndcg"]} for k in k_values}
    return averages
 
 
if __name__ == "__main__":
    qrels = pd.read_csv(QRELS_CSV)
    label_map = {"E": 3, "S": 2, "C": 1, "I": 0}
    qrels["relevance"] = qrels["esci_label"].map(label_map)
 
    print(f"Total filas en qrels: {len(qrels)}")
    print(f"Queries únicas a evaluar: {qrels['query'].nunique()}")
 
    k_values = [1, 3, 5, 10]
    avg = evaluate(qrels, k_values)
 
    print("\n=== RESULTADOS DE EVALUACIÓN ===")
    print("k\tPrecision@k\tRecall@k\tNDCG@k")
    for k in k_values:
        print(f"{k}\t{avg[k]['precision']:.4f}\t\t{avg[k]['recall']:.4f}\t\t{avg[k]['ndcg']:.4f}")