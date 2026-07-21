"""
Sistema RAG multimodal con ChromaDB y OpenRouter (modelos gratuitos)
"""
import base64
import mimetypes
import os
from openai import OpenAI
from dotenv import load_dotenv
from retrieve import search
 
# Cargar variables de entorno
load_dotenv()
 
# Cliente de OpenRouter
api_key = os.getenv("OpenRouter_API_KEY") # os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("OPENROUTER_API_KEY no encontrada.")
 
client = OpenAI(
    # base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

def generate_response(query, docs, query_type="text", max_images=4):
    """
    Genera respuesta usando Gemini con contexto de los productos recuperados.
    Si query_type == "image", `query` es la ruta a la imagen de referencia
    y se envía junto con las imágenes de los productos recuperados.
    """
    if not docs:
        return "No se encontraron productos relevantes para tu consulta."

    if query_type == "image":
        user_question_text = (
            "El usuario subió la imagen de referencia adjunta al inicio. "
            "Las imágenes siguientes corresponden a los productos más similares "
            "encontrados por el sistema (en el mismo orden que el contexto de texto). "
            "Descríbelos, explica visualmente por qué se parecen a la referencia "
            "y destaca sus características principales."
        )
    else:
        user_question_text = query

    # --- Contexto de texto ---
    context_parts = []
    for i, doc in enumerate(docs, 1):
        descripcion = doc.get('descripcion', '')
        if not descripcion or str(descripcion).lower() == "nan":
            descripcion = "Sin descripción"
        text = f"Producto {i}: {doc.get('titulo', 'Sin título')}\nDescripción: {descripcion}"
        context_parts.append(text)
    context = "\n\n".join(context_parts)

    prompt = f"""Eres un asistente de compras experto. Responde la pregunta del usuario basándote ÚNICAMENTE en el contexto de productos proporcionado (texto e imágenes).

REGLAS IMPORTANTES:
1. Usa SOLO la información del contexto (texto + imágenes adjuntas).
2. Si el contexto no contiene suficiente información para responder, di: "No tengo suficiente información en el corpus para responder esta pregunta."
3. Sé conciso pero informativo, describiendo los productos en detalle, incluyendo lo que ves en sus imágenes.
4. Menciona los productos específicos (por su número "Producto N") que respaldan tu respuesta.

### Contexto de productos:
{context}

### Pregunta del usuario:
{user_question_text}

### Respuesta:
"""

    content_blocks = [{"type": "text", "text": prompt}]

    # 1) Imagen de referencia del usuario (solo si query_type == "image")
    if query_type == "image":
        ref = resolve_image_url(query)
        if ref:
            content_blocks.insert(0, {"type": "image_url", "image_url": {"url": ref}})

    # 2) Imágenes de los productos recuperados (limitadas)
    images_added = 0
    for doc in docs:
        if images_added >= max_images:
            break
        img_url = resolve_image_url(doc.get('image_path'))
        if img_url:
            content_blocks.append({"type": "image_url", "image_url": {"url": img_url}})
            images_added += 1

    try:
        response = client.chat.completions.create(
            model="openrouter/free",
            messages=[{"role": "user", "content": content_blocks}],
            temperature=0.3,
            max_tokens=800
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error al generar respuesta: {e}"


def resolve_image_url(image_path):
    """Devuelve una URL usable por el modelo: pasa URLs http(s) directo,
    o codifica archivos locales en base64 data URI."""
    if not image_path:
        return None
    if isinstance(image_path, str) and image_path.startswith(("http://", "https://")):
        return image_path
    return encode_image(image_path)


def encode_image(image_path):
    if not image_path or not os.path.exists(image_path):
        return None
    try:
        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type:
            mime_type = "image/jpeg"
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{mime_type};base64,{b64}"
    except Exception as e:
        print(f"Error codificando imagen {image_path}: {e}")
        return None
 
def ask(query, query_type="text", top_k=5, text_weight=0.5, image_weight=0.5):
    """
    Pipeline completo: busca y genera respuesta.
    """
    docs = search(query, query_type=query_type, top_k=top_k,
                  text_weight=text_weight, image_weight=image_weight)
    response = generate_response(query, docs, query_type=query_type)
    return response, docs
 
if __name__ == "__main__":
    # Prueba rápida
    print("=== Prueba RAG con OpenRouter ===")
    query = "wireless headphones"
    print(f"\nConsulta: {query}")
    response, docs = ask(query, query_type="text", top_k=5)
    print(f"\nRespuesta:\n{response}")
    print(f"\nProductos recuperados ({len(docs)}):")
    for i, doc in enumerate(docs, 1):
        print(f"  {i}. {doc['titulo']} (score: {doc['score_final']})")
 