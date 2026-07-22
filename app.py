"""
Interfaz conversacional con Streamlit para el sistema RAG multimodal.

Proyecto Final - Sistema de Recuperación de Información Multimodal
Asignatura: Recuperación de Información

Ejecutar:
    streamlit run app.py
"""

import streamlit as st
import os
from rag_system import ask
from retrieve import search  # por si quieres hacer búsquedas directas
from feedback_store import add_feedback, get_stats

# ------------------------------------------------------------------
# Configuración de página
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Sistema RI Multimodal",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# Estilos personalizados
# ------------------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0;
        color: var(--text-color);
    }
    .sub-header {
        color: var(--text-color);
        opacity: 0.65;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }
    .evidence-card {
        border: 1px solid rgba(128, 128, 128, 0.3);
        border-radius: 10px;
        padding: 0.9rem;
        margin-bottom: 0.8rem;
        background-color: var(--secondary-background-color);
    }
    .score-badge {
        display: inline-block;
        background-color: rgba(99, 102, 241, 0.18);
        color: #818cf8;
        font-weight: 600;
        font-size: 0.78rem;
        padding: 2px 10px;
        border-radius: 999px;
        margin-top: 4px;
    }
    .rank-badge {
        display: inline-block;
        background-color: var(--primary-color);
        color: white;
        font-weight: 700;
        font-size: 0.75rem;
        width: 22px;
        height: 22px;
        border-radius: 50%;
        text-align: center;
        line-height: 22px;
        margin-right: 6px;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🔎 Sistema RI Multimodal")
    st.caption("Recuperación de Información + RAG sobre texto e imágenes")

    st.markdown("---")
    st.markdown("#### ⚙️ Parámetros de búsqueda")

    top_k = st.slider("Documentos a recuperar (Top-k)", min_value=1, max_value=20, value=10)

    query_type = st.radio(
        "Tipo de consulta",
        options=["text", "image"],
        format_func=lambda x: "📝 Texto" if x == "text" else "🖼️ Imagen",
        horizontal=True,
        help="Con 'Imagen' podrás subir una foto de referencia en vez de escribir texto.",
    )

    st.markdown("---")
    st.markdown("#### ℹ️ Acerca del sistema")
    st.markdown(
        """
        Este sistema combina:
        - **Embeddings CLIP** multimodales (texto + imagen)
        - **Base de datos vectorial** para indexar y recuperar el corpus
        - **RAG** (Retrieval-Augmented Generation) para generar respuestas
          usando los productos recuperados como contexto
        - **Memoria conversacional** de los últimos turnos
        - **Relevance Feedback** (👍/👎) que ajusta futuras búsquedas
        """
    )

    st.markdown("---")
    if st.button("🗑️ Limpiar conversación", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ------------------------------------------------------------------
# Encabezado principal
# ------------------------------------------------------------------
st.markdown('<p class="main-header">Sistema de Recuperación de Información Multimodal</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Consulta productos en lenguaje natural — el sistema recupera evidencia '
    'relevante (texto + imágenes) y genera una respuesta a partir de ese contexto.</p>',
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# Utilidad para mostrar evidencias
# ------------------------------------------------------------------
def render_evidences(docs, msg_key):
    """
    msg_key: identificador único del mensaje (para que las keys de los
    botones de feedback no choquen entre distintos mensajes del chat).
    """
    if not docs:
        return
    with st.expander(f"📦 Ver evidencia utilizada ({len(docs)} productos)", expanded=False):
        for i, doc in enumerate(docs):
            pid = doc.get("product_id", f"idx{i}")
            with st.container():
                st.markdown('<div class="evidence-card">', unsafe_allow_html=True)
                col1, col2, col3 = st.columns([1, 3, 1])
                with col1:
                    img_path = doc.get("image_path", "")
                    if img_path and os.path.exists(img_path):
                        st.image(img_path, use_container_width=True)
                    else:
                        st.markdown(
                            "<div style='text-align:center; padding-top:20px;'>🖼️<br>"
                            "<small>Sin imagen</small></div>", unsafe_allow_html=True
                        )
                with col2:
                    st.markdown(
                        f'<span class="rank-badge">{i+1}</span>'
                        f'**{doc.get("titulo", "Sin título")}**',
                        unsafe_allow_html=True,
                    )
                    score = doc.get("score_final", 0.0)
                    st.markdown(f'<span class="score-badge">score: {score:.4f}</span>', unsafe_allow_html=True)
                    descripcion = doc.get("descripcion", "")
                    if descripcion and str(descripcion).lower() != "nan":
                        st.write(str(descripcion)[:180] + "...")
                with col3:
                    stats = get_stats(pid)
                    btn_col1, btn_col2 = st.columns(2)
                    with btn_col1:
                        if st.button("👍", key=f"like_{msg_key}_{pid}", help="Este resultado fue útil"):
                            add_feedback(pid, liked=True)
                            st.toast(f"Gracias por tu feedback sobre '{doc.get('titulo', '')[:30]}'")
                            st.rerun()
                    with btn_col2:
                        if st.button("👎", key=f"dislike_{msg_key}_{pid}", help="Este resultado no fue útil"):
                            add_feedback(pid, liked=False)
                            st.toast("Feedback registrado, ajustaremos futuras búsquedas")
                            st.rerun()
                    st.caption(f"👍{stats['likes']} · 👎{stats['dislikes']}")
                st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------
# Estado / historial de la conversación
# ------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar mensajes anteriores
for msg_idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg.get("user_image"):
            st.image(msg["user_image"], width=180)
        st.markdown(msg["content"])
        if msg.get("evidences"):
            render_evidences(msg["evidences"], msg_key=f"hist{msg_idx}")

# ------------------------------------------------------------------
# Input del usuario (texto o imagen, según el modo elegido en el sidebar)
# ------------------------------------------------------------------
prompt = None
image_query_path = None
display_content = None

if query_type == "text":
    prompt = st.chat_input("Escribe tu consulta sobre productos...")
    if prompt:
        display_content = prompt

else:  # query_type == "image"
    uploaded_file = st.file_uploader(
        "Sube una imagen de referencia (ej. una foto del producto que buscas)",
        type=["jpg", "jpeg", "png"],
    )
    if uploaded_file is not None:
        st.image(uploaded_file, width=180, caption="Imagen a buscar")
        if st.button("🔍 Buscar productos similares", type="primary"):
            os.makedirs("data/tmp_uploads", exist_ok=True)
            # nombre único para no pisar archivos de otras búsquedas en la misma sesión
            temp_path = os.path.join(
                "data/tmp_uploads",
                f"{len(st.session_state.messages)}_{uploaded_file.name}"
            )
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            image_query_path = temp_path
            display_content = f"🖼️ Imagen subida: {uploaded_file.name}"

# ------------------------------------------------------------------
# Memoria conversacional básica
# ------------------------------------------------------------------
def build_memoria_context(messages, max_turns=3):
    """
    Devuelve un string con los últimos `max_turns` pares usuario/asistente,
    para dárselo al LLM como contexto de la conversación.

    Importante: este texto SOLO se usa para la generación de la respuesta
    (se lo pasamos a rag_system.ask() como `memory`). NUNCA se mezcla con
    la query real que se embeddea para la búsqueda vectorial -- si se
    metiera ahí, además de romper la búsqueda por imagen (la query dejaría
    de ser una ruta de archivo), degradaría la calidad del embedding de
    texto, ya que CLIP está pensado para captions cortas, no para
    historiales de chat completos.
    """
    if not messages:
        return ""
    memo = []
    relevant_msgs = messages[-(max_turns * 2):]
    for m in relevant_msgs:
        if m["role"] == "user":
            memo.append(f"Usuario: {m['content']}")
        elif m["role"] == "assistant":
            memo.append(f"Asistente: {m['content']}")
    return "\n".join(memo)


if prompt or image_query_path:
    # La memoria se calcula ANTES de agregar el turno actual (para no
    # incluirse a sí misma), y se usa solo para la generación de respuesta.
    memoria_context = build_memoria_context(st.session_state.messages, max_turns=3)

    # La query real para búsqueda es SIEMPRE el input actual (texto o ruta
    # de imagen), sin mezclar memoria -- eso se pasa aparte a ask().
    actual_query = prompt if query_type == "text" else image_query_path

    # Mensaje del usuario
    st.session_state.messages.append({
        "role": "user",
        "content": display_content,
        "user_image": image_query_path if query_type == "image" else None,
    })
    with st.chat_message("user"):
        if image_query_path:
            st.image(image_query_path, width=180)
        st.markdown(display_content)

    with st.chat_message("assistant"):
        with st.spinner("Recuperando evidencia y generando respuesta..."):
            try:
                response, docs = ask(
                    actual_query,
                    query_type=query_type,
                    top_k=top_k,
                    memory=memoria_context,
                )
            except Exception as e:
                response = f"⚠️ Ocurrió un error al procesar la consulta: {e}"
                docs = []

        # Se guarda en el historial ANTES de renderizar los botones de feedback:
        # si el usuario da clic en 👍/👎, dispara un st.rerun(), y si el mensaje
        # no estuviera ya guardado se perdería de la conversación.
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "evidences": docs,
        })
        new_msg_idx = len(st.session_state.messages) - 1

        st.markdown(response)
        render_evidences(docs, msg_key=f"hist{new_msg_idx}")