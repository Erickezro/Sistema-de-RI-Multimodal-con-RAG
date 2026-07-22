"""
Descarga y unión del corpus multimodal:
  - ESCI Shopping Queries Dataset (texto + etiquetas de relevancia)
  - SQID (imágenes)

Requisitos:
    pip install datasets pandas pillow requests tqdm pyarrow huggingface_hub

Uso:
    python download_corpus.py
"""

import logging
import os
import pandas as pd
import requests
from datasets import load_dataset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/raw/images", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# Descargar ESCI (texto + esci_label) desde Hugging Face
print("Descargando ESCI (queries + products) desde Hugging Face...")
esci_queries = load_dataset("milistu/amazon-esci-data", name="queries", split="test")
esci_products = load_dataset("milistu/amazon-esci-data", name="products", split="test")

df_examples = esci_queries.to_pandas()
df_products = esci_products.to_pandas()

print(f"examples: {df_examples.shape}, columnas: {list(df_examples.columns)}")
print(f"products: {df_products.shape}, columnas: {list(df_products.columns)}")


df_examples = df_examples[df_examples["product_locale"] == "us"]
if "small_version" in df_examples.columns:
    df_examples = df_examples[df_examples["small_version"] == 1]

df_text = df_examples.merge(
    df_products, how="left", on=["product_id", "product_locale"]
)
print(f"Después del merge examples+products: {df_text.shape}")

# ------------------------------------------------------------------
# Descargar SQID: product_id -> image_url
#    (subset "product_image_urls" es el que trae la relación producto-imagen)
# ------------------------------------------------------------------
print("Descargando SQID desde Hugging Face...")
sqid_images = load_dataset(
    "crossingminds/shopping-queries-image-dataset",
    name="product_image_urls",
    split="train",
)
df_images = sqid_images.to_pandas()
df_images = df_images[df_images["image_url"].notna()]
print(f"SQID con imagen válida: {df_images.shape}")


# Unir texto + imagen por product_id
df_full = df_text.merge(df_images, how="inner", on="product_id")
print(f"Corpus completo (texto + imagen): {df_full.shape}")
print(df_full[["query", "product_title", "esci_label", "image_url"]].head())


# Tomamos un subconjunto manejable
N_PRODUCTS = 2000
# Priorizamos tener consultas completas (todos los productos juzgados de
# algunas queries) en vez de un sample aleatorio de filas sueltas
sample_queries = df_full["query_id"].drop_duplicates().sample(
    n=min(300, df_full["query_id"].nunique()), random_state=42
)
df_sub = df_full[df_full["query_id"].isin(sample_queries)].head(N_PRODUCTS).reset_index(drop=True)
print(f"\nSubconjunto final: {len(df_sub)} filas, {df_sub['query_id'].nunique()} queries únicas")


# Descargar imágenes localmente
records = []
n_failed = 0
for i, row in tqdm(df_sub.iterrows(), total=len(df_sub), desc="Descargando imágenes"):
    img_path = f"data/raw/images/{row['product_id']}.jpg"
    if not os.path.exists(img_path):
        try:
            resp = requests.get(row["image_url"], timeout=10)
            resp.raise_for_status()
            with open(img_path, "wb") as f:
                f.write(resp.content)
        except (requests.RequestException, OSError) as e:
            # Una descarga fallida NO debe abortar todo el corpus, pero
            # tampoco debe desaparecer sin dejar rastro: la registramos.
            n_failed += 1
            logger.warning("No se pudo descargar la imagen de %s (%s): %s",
                           row["product_id"], row["image_url"], e)
            continue

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

# Guardamos el corpus procesado
corpus_df = pd.DataFrame(records)
corpus_df.to_csv("data/processed/corpus.csv", index=False)
print(f"\nCorpus final guardado en data/processed/corpus.csv ({len(corpus_df)} filas)")
if n_failed:
    logger.warning("%d imágenes no pudieron descargarse y se omitieron del corpus.", n_failed)