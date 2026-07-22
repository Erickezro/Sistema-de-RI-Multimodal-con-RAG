# ============================================================
# CARGA DE EMBEDDINGS Y METADATOS
# ============================================================
import pandas as pd
import numpy as np
import chromadb

# Cargamos los IDs de los productos (en el mismo orden en que se generaron los embeddings)
ids_df = pd.read_csv("data/processed/product_embedding_ids.csv")
product_ids = ids_df["product_id"].astype(str).tolist()

# Cargamos los embeddings de texto e imagen (arrays NumPy)
text_embs = np.load("data/processed/product_embeddings_text.npy")
image_embs = np.load("data/processed/product_embeddings_image.npy")

# Verificación de seguridad: todos deben tener la misma cantidad de elementos
assert len(product_ids) == len(text_embs) == len(image_embs), \
    "Los IDs y los embeddings no tienen el mismo largo -- algo quedó desalineado"

# Cargamos la tabla de productos para obtener metadatos adicionales
products = pd.read_csv("data/processed/products.csv").set_index("product_id")

# ============================================================
# CONEXIÓN A CHROMADB Y CREACIÓN DE COLECCIONES
# ============================================================
# ChromaDB es una base de datos vectorial que almacena embeddings y permite
# búsquedas por similitud. Usamos una instancia persistente en disco.
chroma_client = chromadb.PersistentClient(path="data/chroma_db")

# Creamos dos colecciones: una para embeddings de texto, otra para imagen.
# Ambas usan distancia coseno (porque nuestros embeddings están normalizados).
text_collection = chroma_client.get_or_create_collection(
    name="products_text", metadata={"hnsw:space": "cosine"}
)
image_collection = chroma_client.get_or_create_collection(
    name="products_image", metadata={"hnsw:space": "cosine"}
)

# ============================================================
# PREPARACIÓN DE METADATOS Y DOCUMENTOS PARA CADA COLECCIÓN
# ============================================================
# ChromaDB permite guardar metadatos (ej. título, descripción) y documentos
# (texto asociado) junto con los embeddings para recuperarlos después.
metadatas, documents_text, documents_image = [], [], []
for pid in product_ids:
    row = products.loc[pid]
    metadatas.append({
        "product_id": pid,
        "title": str(row["product_title"]),
        "description": str(row["product_description"]),
        "image_path": str(row["image_path"]),
    })
    # Para la colección de texto, el documento es el título + descripción
    documents_text.append(f"{row['product_title']} {row['product_description']}")
    # Para la colección de imagen, guardamos la ruta de la imagen (como referencia)
    documents_image.append(str(row["image_path"]))

# ============================================================
# INSERCIÓN DE DATOS EN LAS COLECCIONES (SOLO SI ESTÁN VACÍAS)
# ============================================================
# Insertamos solo si la colección está vacía, para evitar duplicar datos
# si ejecutamos el script varias veces.
if text_collection.count() == 0:
    text_collection.add(
        ids=product_ids, embeddings=text_embs.tolist(),
        documents=documents_text, metadatas=metadatas,
    )
if image_collection.count() == 0:
    image_collection.add(
        ids=product_ids, embeddings=image_embs.tolist(),
        documents=documents_image, metadatas=metadatas,
    )

# Mostramos cuántos documentos hay en cada colección (debería coincidir con el total de productos)
print(f"Texto: {text_collection.count()} documentos")
print(f"Imagen: {image_collection.count()} documentos")