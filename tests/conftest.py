"""
Configuración compartida de los tests.

Varios módulos del proyecto ejecutan efectos secundarios pesados al
importarse (cargar el modelo CLIP con torch, abrir ChromaDB, leer CSVs
del corpus, instanciar un Cross-Encoder, etc.) o requieren una API key.
Nada de eso es necesario para probar la LÓGICA pura de los módulos, así
que aquí:

  * se agrega la raíz del repo al `sys.path` para poder importar los
    módulos por su nombre (``import feedback_store``);
  * se inyectan módulos falsos ("stubs") en ``sys.modules`` para las
    dependencias pesadas que no queremos instalar ni ejecutar de verdad
    (``sentence_transformers`` y el módulo ``retrieve``, que a su vez
    arrastra torch/ChromaDB y lee el corpus);
  * se define una API key de prueba para que ``rag_system`` y
    ``query_expansion`` puedan importarse sin un ``.env`` real.

Estos stubs solo cubren dependencias que los tests no ejercitan
directamente: la lógica que sí se prueba es siempre la del módulo real.
"""

import os
import sys
import types

# Raíz del repositorio (carpeta que contiene este directorio tests/).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# API key de prueba: rag_system aborta el import si no encuentra ninguna,
# y query_expansion decide si crear su cliente en función de ella.
os.environ.setdefault("OPENROUTER_API_KEY", "test-key-not-real")


def _install_fake_sentence_transformers():
    """Stub de sentence_transformers.CrossEncoder (no está instalado)."""
    if "sentence_transformers" in sys.modules:
        return
    mod = types.ModuleType("sentence_transformers")

    class CrossEncoder:  # pragma: no cover - stub trivial
        def __init__(self, *args, **kwargs):
            pass

        def predict(self, pairs):
            return [0.0 for _ in pairs]

    mod.CrossEncoder = CrossEncoder
    sys.modules["sentence_transformers"] = mod


def _install_fake_retrieve():
    """Stub del módulo retrieve (evita torch, ChromaDB y leer el corpus)."""
    if "retrieve" in sys.modules:
        return
    mod = types.ModuleType("retrieve")

    def search(*args, **kwargs):  # pragma: no cover - los tests lo sustituyen
        return []

    mod.search = search
    sys.modules["retrieve"] = mod


_install_fake_sentence_transformers()
_install_fake_retrieve()
