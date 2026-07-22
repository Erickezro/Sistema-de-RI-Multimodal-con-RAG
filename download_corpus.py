"""
Descarga y unión del corpus multimodal:
  - ESCI Shopping Queries Dataset (texto + etiquetas de relevancia)
  - SQID (imágenes)

Requisitos:
    pip install datasets pandas pillow requests tqdm pyarrow huggingface_hub

Uso:
    python download_corpus.py
"""

import os
import pandas as pd
import requests
from datasets import load_dataset
from tqdm import tqdm

# ============================================================
# CREACIÓN DE CARPETAS NECESARIAS
# ============================================================
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/raw/images", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# ============================================================
# DESCARGA DEL DATASET ESCI (queries y productos)
# ============================================================
print("Descargando ESCI (queries + products) desde Hugging Face...")
esci_queries = load_dataset("milistu/amazon-esci-data", name="queries", split="test")
esci_products = load_dataset("milistu/amazon-esci-data", name="products", split="test")

df_examples = esci_queries.to_pandas()   # Contiene las queries y los pares query-producto
df_products = esci_products.to_pandas()  # Contiene los metadatos de los productos

print(f"examples: {df_examples.shape}, columnas: {list(df_examples.columns)}")
print(f"products: {df_products.shape}, columnas: {list(df_products.columns)}")

# Filtramos solo productos en idioma inglés (locale = 'us')
df_examples = df_examples[df_examples["product_locale"] == "us"]
# Si existe la columna 'small_version', nos quedamos con la versión pequeña (datos más ligeros)
if "small_version" in df_examples.columns:
    df_examples = df_examples[df_examples["small_version"] == 1]

# Unimos las tablas de queries y productos por product_id y locale
df_text = df_examples.merge(
    df_products, how="left", on=["product_id", "product_locale"]
)
print(f"Después del merge examples+products: {df_text.shape}")

# ============================================================
# DESCARGA DEL DATASET SQID (imágenes)
# ============================================================
print("Descargando SQID desde Hugging Face...")
sqid_images = load_dataset(
    "crossingminds/shopping-queries-image-dataset",
    name="product_image_urls",
    split="train",
)
df_images = sqid_images.to_pandas()
df_images = df_images[df_images["image_url"].notna()]   # Solo filas con URL de imagen
print(f"SQID con imagen válida: {df_images.shape}")

# ============================================================
# UNIÓN DE DATOS DE TEXTO E IMAGEN
# ============================================================
# Hacemos un inner join para quedarnos solo con productos que tienen imagen
df_full = df_text.merge(df_images, how="inner", on="product_id")
print(f"Corpus completo (texto + imagen): {df_full.shape}")
print(df_full[["query", "product_title", "esci_label", "image_url"]].head())

# ============================================================
# SELECCIÓN DE UN SUBCONJUNTO MANEJABLE
# ============================================================
N_PRODUCTS = 2000
# Para tener queries completas (con todos sus productos juzgados), 
# seleccionamos un conjunto de queries al azar y luego tomamos hasta N_PRODUCTS filas.
sample_queries = df_full["query_id"].drop_duplicates().sample(
    n=min(300, df_full["query_id"].nunique()), random_state=42
)
df_sub = df_full[df_full["query_id"].isin(sample_queries)].head(N_PRODUCTS).reset_index(drop=True)
print(f"\nSubconjunto final: {len(df_sub)} filas, {df_sub['query_id'].nunique()} queries únicas")

# ============================================================
# DESCARGA LOCAL DE LAS IMÁGENES
# ============================================================
records = []
for i, row in tqdm(df_sub.iterrows(), total=len(df_sub), desc="Descargando imágenes"):
    img_path = f"data/raw/images/{row['product_id']}.jpg"
    # Solo descargamos si no existe ya
    if not os.path.exists(img_path):
        try:
            resp = requests.get(row["image_url"], timeout=10)
            resp.raise_for_status()
            with open(img_path, "wb") as f:
                f.write(resp.content)
        except Exception:
            continue   # Si falla, saltamos esta imagen

    # Si la imagen se descargó correctamente (o ya existía), guardamos el registro
    if os.path.exists(img_path):
        records.append({
            "product_id": row["product_id"],
            "query_id": row["query_id"],
            "query": row["query"],
            "product_title": row["product_title"],
            "product_description": row.get("product_description", ""),
            "esci_label": row["esci_label"],
            "image_path": img_path,
        })

# ============================================================
# GUARDADO DEL CORPUS PROCESADO
# ============================================================
corpus_df = pd.DataFrame(records)
corpus_df.to_csv("data/processed/corpus.csv", index=False)
print(f"\nCorpus final guardado en data/processed/corpus.csv ({len(corpus_df)} filas)")