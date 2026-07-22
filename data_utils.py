"""
Utilidades de datos compartidas.

Centraliza las rutas de los archivos del corpus y la carga de la tabla
de productos, que antes se repetían (con la misma ruta hard-coded) en
retrieve.py, vector_db.py y generate_embeddings.py. También expone un
helper para normalizar descripciones: el corpus guarda las descripciones
vacías como el texto 'nan' al serializar NaN a CSV, y varios módulos lo
limpiaban por su cuenta.
"""
import os

import pandas as pd

DATA_DIR = "data"
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
RAW_DIR = os.path.join(DATA_DIR, "raw")
RAW_IMAGES_DIR = os.path.join(RAW_DIR, "images")

PRODUCTS_CSV = os.path.join(PROCESSED_DIR, "products.csv")
CORPUS_CSV = os.path.join(PROCESSED_DIR, "corpus.csv")
QRELS_CSV = os.path.join(PROCESSED_DIR, "qrels.csv")
EMBEDDINGS_TEXT_NPY = os.path.join(PROCESSED_DIR, "product_embeddings_text.npy")
EMBEDDINGS_IMAGE_NPY = os.path.join(PROCESSED_DIR, "product_embeddings_image.npy")
EMBEDDING_IDS_CSV = os.path.join(PROCESSED_DIR, "product_embedding_ids.csv")

CHROMA_DB_DIR = os.path.join(DATA_DIR, "chroma_db")
FEEDBACK_PATH = os.path.join(DATA_DIR, "feedback.json")


def load_products(index_by_id=False):
    """Carga la tabla de productos del corpus procesado.

    index_by_id=True: devuelve el DataFrame indexado por product_id, para
    el acceso por .loc[pid] que usan retrieve.py y vector_db.py.
    """
    df = pd.read_csv(PRODUCTS_CSV)
    if index_by_id:
        df = df.set_index("product_id")
    return df


def clean_description(value, default=""):
    """Normaliza una descripción de producto.

    Devuelve `default` si el valor está vacío o si es el string 'nan'
    (el corpus guarda las descripciones vacías como el texto 'nan' al
    serializar NaN a CSV); en caso contrario devuelve el texto tal cual.
    """
    if value is None:
        return default
    text = str(value)
    if not text or text.lower() == "nan":
        return default
    return text
