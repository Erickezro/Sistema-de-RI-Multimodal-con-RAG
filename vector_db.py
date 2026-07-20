import pandas as pd
import numpy as np
import chromadb

ids_df = pd.read_csv("data/processed/product_embedding_ids.csv")
text_embs = np.load("data/processed/product_embeddings_text.npy")
image_embs = np.load("data/processed/product_embeddings_image.npy")
products = pd.read_csv("data/processed/products.csv").set_index("product_id")

product_ids = ids_df["product_id"].astype(str).tolist()

client = chromadb.PersistentClient(path="data/chroma_db")

