"""
Cliente LLM compartido (OpenRouter).

Centraliza lo que antes repetían rag_system.py y query_expansion.py:
  - la resolución de la API key (aceptando ambas variantes de nombre),
  - la creación del cliente OpenAI apuntando a OpenRouter,
  - la lista de modelos gratuitos usados en el proyecto,
  - la lógica de fallback que prueba varios modelos en cadena.
"""
import os

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Modelos gratuitos con soporte de visión, en orden de preferencia.
# Los ":free" de OpenRouter a veces se saturan temporalmente (rate limit
# del proveedor detrás, ej. Google AI Studio) -- se prueban varios en
# cadena para que una demo en vivo no dependa de un solo proveedor caído.
# Se pone primero el de NVIDIA porque corre en un proveedor distinto a
# Google AI Studio, con un cupo de rate limit independiente.
FREE_MODELS = [
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    "google/gemma-4-26b-a4b-it:free",
    "google/gemma-4-31b-it:free",
]


def get_api_key():
    """Resuelve la API key de OpenRouter aceptando ambas variantes de
    nombre, ya que el .env puede tener cualquiera según quién lo configuró."""
    return os.getenv("OPENROUTER_API_KEY") or os.getenv("OpenRouter_API_KEY")


def get_client(required=True):
    """Devuelve un cliente OpenAI apuntando a OpenRouter.

    required=True (por defecto): lanza ValueError si falta la API key.
    required=False: devuelve None si falta, para los módulos que se
    degradan de forma segura sin la key (ej. query_expansion).
    """
    api_key = get_api_key()
    if not api_key:
        if required:
            raise ValueError(
                "OPENROUTER_API_KEY no encontrada. Revisa tu archivo .env."
            )
        return None
    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)


def chat_with_fallback(client, messages, models=None, on_error=None, **kwargs):
    """
    Llama a chat.completions.create probando `models` en orden hasta que
    uno responda. Devuelve una tupla (content, error):
      - content: el texto de la respuesta, o None si todos fallaron.
      - error: la última excepción capturada, o None si hubo éxito.

    on_error(model_name, exc): callback opcional para loguear cada fallo.
    Los kwargs extra (temperature, max_tokens, ...) se pasan tal cual a la API.
    """
    if models is None:
        models = FREE_MODELS
    last_error = None
    for model_name in models:
        try:
            response = client.chat.completions.create(
                model=model_name, messages=messages, **kwargs
            )
            return response.choices[0].message.content, None
        except Exception as e:
            last_error = e
            if on_error is not None:
                on_error(model_name, e)
            continue
    return None, last_error
