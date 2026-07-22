"""
Query Expansion

Genera reformulaciones/variantes de la consulta original usando un LLM
(sinónimos, términos relacionados, reformulaciones), para buscar con
varias formulaciones y combinar los resultados. Esto ayuda a recuperar
documentos relevantes que usan vocabulario distinto al de la consulta
original del usuario (ej. "audífonos" vs "auriculares" vs "earbuds").

Diseño defensivo: si la llamada al LLM falla (rate limit, sin conexión,
respuesta rara, etc.) se degrada de forma segura devolviendo una lista
vacía -- quien llama a expand_query() simplemente sigue con la consulta
original sin expansión, sin que la búsqueda se rompa.
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

# ============================================================
# CONFIGURACIÓN DEL CLIENTE DE OPENROUTER
# ============================================================
load_dotenv()   # Carga las variables de entorno desde un archivo .env

# Buscamos la API key en las variables de entorno (con dos nombres posibles)
_api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OpenRouter_API_KEY")
# Creamos el cliente de OpenAI apuntando a OpenRouter (un agregador de modelos)
_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=_api_key) if _api_key else None

# ============================================================
# MODELOS DE EXPANSIÓN (orden de preferencia)
# ============================================================
# Modelos de texto livianos para esta tarea (no necesitan soporte de visión).
# Mismos modelos que ya confirmamos funcionando en rag_system.py -- se pone
# primero el de NVIDIA porque corre en un proveedor distinto a Google AI
# Studio, así que tiene un cupo de rate limit independiente si Google está
# saturado (como pasó varias veces durante las pruebas).
EXPANSION_MODELS = [
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    "google/gemma-4-26b-a4b-it:free",
    "google/gemma-4-31b-it:free",
]

# ============================================================
# FUNCIÓN PRINCIPAL: expand_query
# ============================================================
def expand_query(query: str, n_variants: int = 3) -> list:
    """
    Devuelve hasta n_variants reformulaciones de `query` (sin incluir la
    original). Si algo falla, devuelve [] (degradación segura, sin
    excepción propagada).
    """
    # Si no hay cliente o la consulta está vacía, retornamos lista vacía.
    if _client is None or not query or not query.strip():
        return []

    # Construimos el prompt para el LLM.
    prompt = (
        f"Genera {n_variants} reformulaciones o expansiones cortas de esta consulta de "
        f"búsqueda de productos, usando sinónimos y términos relacionados en inglés "
        f"(el catálogo de productos está en inglés). Una por línea, sin numerar, "
        f"sin explicaciones adicionales.\n\nConsulta: \"{query}\""
    )

    # Probamos cada modelo en orden hasta que uno funcione.
    for model_name in EXPANSION_MODELS:
        try:
            # Llamada a la API de OpenRouter.
            response = _client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,          # Cierta creatividad, pero sin disparates
                max_tokens=100,           # Respuestas cortas
            )
            text = (response.choices[0].message.content or "").strip()
            # Dividimos por líneas y limpiamos caracteres de viñetas/números.
            variants = [line.strip("-•*0123456789. \t") for line in text.split("\n")]
            # Filtramos líneas vacías o que sean igual a la consulta original.
            variants = [v for v in variants if v and v.lower() != query.strip().lower()]
            if variants:
                return variants[:n_variants]   # Devolvemos las primeras n variantes
        except Exception as e:
            print(f"[DEBUG expand_query] Falló {model_name}: {e}")
            continue  # Pasamos al siguiente modelo

    # Si todos los modelos fallaron, mostramos un mensaje (opcional) y devolvemos lista vacía.
    if _client is not None:
        print("[DEBUG expand_query] Todos los modelos fallaron o devolvieron vacío.")

    return []   # Degradación segura: sin expansión