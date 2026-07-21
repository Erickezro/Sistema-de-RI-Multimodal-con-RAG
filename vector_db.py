import pandas as pd
import numpy as np
import chromadb


ids_df = pd.read_csv("data/processed/product_embedding_ids.csv")
product_ids = ids_df["product_id"].astype(str).tolist()

text_embs = np.load("data/processed/product_embeddings_text.npy")
image_embs = np.load("data/processed/product_embeddings_image.npy")

assert len(product_ids) == len(text_embs) == len(image_embs), \
    "Los IDs y los embeddings no tienen el mismo largo -- algo quedó desalineado"

products = pd.read_csv("data/processed/products.csv").set_index("product_id")

chroma_client = chromadb.PersistentClient(path="data/chroma_db")

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