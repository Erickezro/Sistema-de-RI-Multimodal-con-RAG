"""
Etapa 2 - Generación de embeddings con CLIP

Genera un embedding multimodal por producto (combinando imagen + texto)
y una función reutilizable para embeddear queries del usuario.

Requisitos:
    pip install open_clip_torch torch pillow pandas numpy tqdm

Uso:
    python generate_embeddings.py
"""

import logging

import torch
import open_clip
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# Cargar el modelo CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"

model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
tokenizer = open_clip.get_tokenizer("ViT-B-32")
model = model.to(device).eval()

print(f"Modelo CLIP cargado en: {device}")


# Funciones de embedding (texto e imagen)
@torch.no_grad()
def embed_text(text: str):
    tokens = tokenizer([text]).to(device)
    embedding = model.encode_text(tokens)
    embedding = embedding / embedding.norm(dim=-1, keepdim=True)   # normalizar
    return embedding.cpu().numpy()[0]


@torch.no_grad()
def embed_image(image_path: str):
    img = Image.open(image_path).convert("RGB")
    img_t = preprocess(img).unsqueeze(0).to(device)
    embedding = model.encode_image(img_t)
    embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return embedding.cpu().numpy()[0]


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Generar embeddings para todos los productos del corpus
    # ------------------------------------------------------------------
    products = pd.read_csv("data/processed/products.csv")

    product_ids = []
    text_embeddings = []
    image_embeddings = []

    for _, row in tqdm(products.iterrows(), total=len(products), desc="Embeddings de productos"):
        try:
            text = row["text"]
            text_vec = embed_text(text)
            img_vec = embed_image(row["image_path"])

            product_ids.append(row["product_id"])
            text_embeddings.append(text_vec)
            image_embeddings.append(img_vec)
        except Exception as e:
            logger.warning("Falló el embedding de %s: %s", row["product_id"], e)
            continue

    n_failed = len(products) - len(product_ids)
    if n_failed:
        logger.warning("%d de %d productos no se pudieron embeddear.", n_failed, len(products))
    if not product_ids:
        raise RuntimeError(
            "Ningún producto pudo embeddearse; revisa data/processed/products.csv "
            "y las rutas de imágenes (data/raw/images). No hay nada que guardar."
        )

    text_embeddings = np.vstack(text_embeddings)
    image_embeddings = np.vstack(image_embeddings)
    print(f"\nEmbeddings de texto: {text_embeddings.shape}")
    print(f"Embeddings de imagen: {image_embeddings.shape}")

    # ------------------------------------------------------------------
    #  Guardar en disco
    # ------------------------------------------------------------------
    np.save("data/processed/product_embeddings_text.npy", text_embeddings)
    np.save("data/processed/product_embeddings_image.npy", image_embeddings)
    pd.DataFrame({"product_id": product_ids}).to_csv(
        "data/processed/product_embedding_ids.csv", index=False
    )
    print("Guardado: product_embeddings_text.npy, product_embeddings_image.npy, "
          "product_embedding_ids.csv")

    print(len(product_ids))
    print(text_embeddings.shape)
    print(image_embeddings.shape)

    print(np.linalg.norm(text_embeddings[0]))
    print(np.linalg.norm(image_embeddings[0]))