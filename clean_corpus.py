"""
Limpieza y estructuración del corpus

Toma data/processed/corpus.csv (salida del download_corpus.py) y genera:
    - data/processed/products.csv -> productos únicos (para indexar/embeddings)
    - data/processed/qrels.csv -> juicios de relevancia (para la evaluación)

Uso:
    python clean_corpus.py
"""

import pandas as pd
from PIL import Image
import os
from bs4 import BeautifulSoup
import re


# ============================================================
# 1. CARGA DEL CORPUS ORIGINAL
# ============================================================
df = pd.read_csv("data/processed/corpus.csv")
print(f"Filas originales: {len(df)}")


# ============================================================
# 2. VALIDACIÓN DE IMÁGENES (descartar corruptas o inexistentes)
# ============================================================
def is_valid_image(path):
    """Intenta abrir la imagen y verificar que no esté corrupta."""
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False

print("Validando imágenes...")
df["image_valid"] = df["image_path"].apply(
    lambda p: os.path.exists(p) and is_valid_image(p)
)
n_invalid = (~df["image_valid"]).sum()
print(f"Imágenes inválidas/corruptas descartadas: {n_invalid}")
df = df[df["image_valid"]].drop(columns=["image_valid"])


# ============================================================
# 3. LIMPIEZA DE TEXTO (eliminar HTML, espacios extra, etc.)
# ============================================================
def clean_html(text):
    """Elimina etiquetas HTML y normaliza espacios."""
    if not isinstance(text, str):
        return ""
    text = BeautifulSoup(text, "html.parser").get_text(" ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

df["product_title"] = df["product_title"].fillna("").apply(clean_html)
df["product_description"] = df["product_description"].fillna("").apply(clean_html)
df["query"] = df["query"].fillna("").str.strip().apply(clean_html)

# Descartar filas sin título (no se puede indexar sin título)
df = df[df["product_title"] != ""]

# Creamos columna 'text' combinando título + descripción (para embeddings)
df["text"] = (df["product_title"].fillna("") + " " + df["product_description"].fillna("")).str.strip()


# ============================================================
# 4. GENERAR products.csv (productos únicos)
# ============================================================
products = (
    df[["product_id", "product_title", "product_description", "text", "image_path"]]
    .drop_duplicates(subset="product_id")
    .reset_index(drop=True)
)
products.to_csv("data/processed/products.csv", index=False)
print(f"\nproducts.csv: {len(products)} productos únicos")


# ============================================================
# 5. GENERAR qrels.csv (juicios de relevancia por consulta)
# ============================================================
qrels = df[["query_id", "query", "product_id", "esci_label"]].drop_duplicates()
qrels.to_csv("data/processed/qrels.csv", index=False)
print(f"qrels.csv: {len(qrels)} juicios de relevancia, "
      f"{qrels['query_id'].nunique()} queries únicas")


# ============================================================
# 6. ESTADÍSTICA FINAL: distribución de etiquetas ESCI
# ============================================================
print("\nDistribución de esci_label:")
print(qrels["esci_label"].value_counts())