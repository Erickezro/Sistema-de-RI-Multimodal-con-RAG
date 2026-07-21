"""
Sistema RAG multimodal con ChromaDB y OpenRouter (modelos gratuitos)
"""

import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from retrieve import search

# Cargar variables de entorno
load_dotenv()

# Configurar cliente de OpenRouter (NO usar OPENAI_API_KEY)
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError("OPENROUTER_API_KEY no encontrada. Crea un archivo .env con ella.")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,  # <--- PASAR EXPLÍCITAMENTE LA CLAVE
)

def generate_response(query, docs, query_type="text"):
    """
    Genera respuesta usando OpenRouter con contexto de los productos recuperados.
    """
    if not docs:
        return "No se encontraron productos relevantes para tu consulta."

    # Construir contexto
    context_parts = []
    for i, doc in enumerate(docs, 1):
        text = f"Producto {i}: {doc['titulo']}\nDescripción: {doc.get('descripcion', 'Sin descripción')}"
        if doc.get('image_path'):
            text += f"\nImagen: {doc['image_path']}"
        context_parts.append(text)
    context = "\n\n".join(context_parts)

    # Prompt
    prompt = f"""Eres un asistente de compras experto. Responde la pregunta del usuario basándote ÚNICAMENTE en el contexto de productos proporcionado.

REGLAS IMPORTANTES:
1. Usa SOLO la información del contexto.
2. Si el contexto no contiene suficiente información para responder, di: "No tengo suficiente información en el corpus para responder esta pregunta."
3. Sé conciso pero informativo.
4. Menciona los productos específicos que respaldan tu respuesta.

### Contexto de productos:
{context}

### Pregunta del usuario:
{query}

### Respuesta:
"""
    try:
        response = client.chat.completions.create(
            model="openrouter/free",  # Enrutador automático a modelos gratuitos  # Modelo gratuito
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error al generar respuesta: {e}"

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