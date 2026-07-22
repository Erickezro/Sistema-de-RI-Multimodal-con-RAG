"""
Etapa 2 - Generación de embeddings con CLIP

Genera un embedding multimodal por producto (combinando imagen + texto)
y una función reutilizable para embeddear queries del usuario.

Requisitos:
    pip install open_clip_torch torch pillow pandas numpy tqdm

Uso:
    python generate_embeddings.py
"""

import torch
import open_clip
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm


# ============================================================
# CARGA DEL MODELO CLIP
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
# Cargamos el modelo ViT-B-32 preentrenado con LAION
model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
tokenizer = open_clip.get_tokenizer("ViT-B-32")
model = model.to(device).eval()

print(f"Modelo CLIP cargado en: {device}")


# ============================================================
# FUNCIONES PARA OBTENER EMBEDDINGS
# ============================================================
@torch.no_grad()
def embed_text(text: str):
    """Genera embedding normalizado para un texto dado."""
    tokens = tokenizer([text]).to(device)
    embedding = model.encode_text(tokens)
    # Normalizamos (L2 norm) para que la similitud sea coseno
    embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return embedding.cpu().numpy()[0]


@torch.no_grad()
def embed_image(image_path: str):
    """Genera embedding normalizado para una imagen dada (ruta)."""
    img = Image.open(image_path).convert("RGB")
    img_t = preprocess(img).unsqueeze(0).to(device)
    embedding = model.encode_image(img_t)
    embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return embedding.cpu().numpy()[0]


# ============================================================
# GENERACIÓN DE EMBEDDINGS PARA TODOS LOS PRODUCTOS
# ============================================================
if __name__ == "__main__":
    # Cargamos el archivo con los productos únicos
    products = pd.read_csv("data/processed/products.csv")

    product_ids = []
    text_embeddings = []
    image_embeddings = []

    # Iteramos sobre cada producto para generar los embeddings
    for _, row in tqdm(products.iterrows(), total=len(products), desc="Embeddings de productos"):
        try:
            text = row["text"]
            text_vec = embed_text(text)
            img_vec = embed_image(row["image_path"])

            product_ids.append(row["product_id"])
            text_embeddings.append(text_vec)
            image_embeddings.append(img_vec)
        except Exception as e:
            print(f"  falló {row['product_id']}: {e}")
            continue

    # Convertimos las listas en arrays numpy (apilados)
    text_embeddings = np.vstack(text_embeddings)
    image_embeddings = np.vstack(image_embeddings)
    print(f"\nEmbeddings de texto: {text_embeddings.shape}")
    print(f"Embeddings de imagen: {image_embeddings.shape}")

    # ============================================================
    # GUARDADO EN DISCO
    # ============================================================
    np.save("data/processed/product_embeddings_text.npy", text_embeddings)
    np.save("data/processed/product_embeddings_image.npy", image_embeddings)
    pd.DataFrame({"product_id": product_ids}).to_csv(
        "data/processed/product_embedding_ids.csv", index=False
    )
    print("Guardado: product_embeddings_text.npy, product_embeddings_image.npy, "
          "product_embedding_ids.csv")

    # Pequeñas comprobaciones de salida
    print(len(product_ids))
    print(text_embeddings.shape)
    print(image_embeddings.shape)
    # Verificamos que están normalizados (norma = 1)
    print(np.linalg.norm(text_embeddings[0]))
    print(np.linalg.norm(image_embeddings[0]))