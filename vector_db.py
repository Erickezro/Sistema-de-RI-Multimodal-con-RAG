import pandas as pd
import numpy as np
import chromadb

from data_utils import (
    load_products,
    EMBEDDING_IDS_CSV,
    EMBEDDINGS_TEXT_NPY,
    EMBEDDINGS_IMAGE_NPY,
    CHROMA_DB_DIR,
)


ids_df = pd.read_csv(EMBEDDING_IDS_CSV)
product_ids = ids_df["product_id"].astype(str).tolist()

text_embs = np.load(EMBEDDINGS_TEXT_NPY)
image_embs = np.load(EMBEDDINGS_IMAGE_NPY)

assert len(product_ids) == len(text_embs) == len(image_embs), \
    "Los IDs y los embeddings no tienen el mismo largo -- algo quedó desalineado"

products = load_products(index_by_id=True)

chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

text_collection = chroma_client.get_or_create_collection(
    name="products_text", metadata={"hnsw:space": "cosine"}
)
image_collection = chroma_client.get_or_create_collection(
    name="products_image", metadata={"hnsw:space": "cosine"}
)


metadatas, documents_text, documents_image = [], [], []
for pid in product_ids:
    row = products.loc[pid]
    metadatas.append({
        "product_id": pid,
        "title": str(row["product_title"]),
        "description": str(row["product_description"]),
        "image_path": str(row["image_path"]),
    })
    documents_text.append(f"{row['product_title']} {row['product_description']}")
    documents_image.append(str(row["image_path"]))

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

print(f"Texto: {text_collection.count()} documentos")
print(f"Imagen: {image_collection.count()} documentos")