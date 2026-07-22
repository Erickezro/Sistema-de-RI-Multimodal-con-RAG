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

from llm_client import get_client, FREE_MODELS

# El cliente puede ser None si falta la API key: expand_query se degrada
# de forma segura en ese caso (ver más abajo).
_client = get_client(required=False)


def expand_query(query: str, n_variants: int = 3) -> list:
    """
    Devuelve hasta n_variants reformulaciones de `query` (sin incluir la
    original). Si algo falla, devuelve [] (degradación segura, sin
    excepción propagada).
    """
    if _client is None or not query or not query.strip():
        return []

    prompt = (
        f"Genera {n_variants} reformulaciones o expansiones cortas de esta consulta de "
        f"búsqueda de productos, usando sinónimos y términos relacionados en inglés "
        f"(el catálogo de productos está en inglés). Una por línea, sin numerar, "
        f"sin explicaciones adicionales.\n\nConsulta: \"{query}\""
    )

    for model_name in FREE_MODELS:
        try:
            response = _client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=100,
            )
            text = (response.choices[0].message.content or "").strip()
            variants = [line.strip("-•*0123456789. \t") for line in text.split("\n")]
            variants = [v for v in variants if v and v.lower() != query.strip().lower()]
            if variants:
                return variants[:n_variants]
        except Exception as e:
            print(f"[DEBUG expand_query] Falló {model_name}: {e}")
            continue  # probar el siguiente modelo de la lista

    if _client is not None:
        print("[DEBUG expand_query] Todos los modelos fallaron o devolvieron vacío.")

    return []