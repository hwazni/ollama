import os
import base64
import streamlit as st
from openai import OpenAI

"""
Streamlit **multimodal chatbot** for vLLM (OpenAI‑compatible) servers
====================================================================

• Behaves like a normal text chatbot.
• If you upload an image, the **very next** user message will include it so the
  model can reason about the picture. Subsequent turns are text‑only.
• Uploading a *different* image automatically starts a **new** conversation to
  avoid the “at most 1 image per request” error.

```bash
pip install streamlit vllm[pillow] "openai>=1.0"
python -m vllm.entrypoints.openai.api_server \
    --model liuhaotian/llava-v1.6-34b \
    --image-input-size 576
```
"""

st.set_page_config(page_title="vLLM Vision Chatbot", layout="centered")
st.title("🖼️📨 vLLM Vision Chatbot")

# ───────────────────────────── Sidebar ──────────────────────────────
with st.sidebar:
    st.header("🔌 Connection Settings")

    server_url = st.text_input("vLLM Server URL", value=os.getenv("VLLM_SERVER_URL", "http://localhost:8000/v1"))
    api_key = st.text_input("API Key (optional)", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    model = st.text_input("Model name on server", value=os.getenv("VLLM_MODEL", "liuhaotian/llava-v1.6-34b"))

    if not server_url:
        st.error("Enter server URL, e.g. http://localhost:8000/v1")
        st.stop()

    client = OpenAI(api_key=api_key or "EMPTY", base_url=server_url.rstrip("/"))

    st.markdown(
        "*Tip*: Upload an image anytime; it will be attached **once** then you can keep chatting. "
        "Uploading a new image resets the conversation so there is never more than one image in a single request.")

# ───────────────────── Helper: image ➜ base64 URL ─────────────────────

def image_to_data_url(file):
    mime_type = file.type
    encoded = base64.b64encode(file.getvalue()).decode()
    return f"data:{mime_type};base64,{encoded}"

# ─────────────────── Session‑state bootstrap ───────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []        # Conversation history
if "pending_image" not in st.session_state:
    st.session_state.pending_image = None # Data URL to send with next user msg
if "last_image_data_url" not in st.session_state:
    st.session_state.last_image_data_url = None  # Tracks current image in session

# ───────────────────────── Image upload ─────────────────────────
uploaded_file = st.file_uploader("Optional image (PNG/JPG/WebP)", type=["png", "jpg", "jpeg", "webp"])
if uploaded_file:
    data_url = image_to_data_url(uploaded_file)

    # Detect a *new* image compared to what was already in context
    if data_url != st.session_state.last_image_data_url:
        # Start a fresh conversation to ensure only one image per request
        st.session_state.messages.clear()
        st.session_state.pending_image = data_url
        st.session_state.last_image_data_url = data_url
    else:
        # Same image re‑selected; don’t reset chat, just preview
        if st.session_state.pending_image is None:  # Already consumed earlier
            st.info("This image is already in context; ask a follow‑up question!")

    st.image(uploaded_file, use_column_width=True)

# ───────────────────────── Chat history render ─────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user" and isinstance(msg["content"], list):
            text = next((c.get("text", "") for c in msg["content"] if c["type"] == "text"), "")
            st.markdown(text)
        else:
            st.markdown(msg["content"])

# ───────────────────────── Chat input ─────────────────────────
user_text = st.chat_input("Message…")
if user_text:
    # Build user content with optional pending image
    if st.session_state.pending_image:
        user_content = [
            {"type": "image_url", "image_url": {"url": st.session_state.pending_image}},
            {"type": "text", "text": user_text},
        ]
        st.session_state.pending_image = None  # Consume image after first use
    else:
        user_content = [{"type": "text", "text": user_text}]

    st.session_state.messages.append({"role": "user", "content": user_content})

    with st.chat_message("user"):
        st.markdown(user_text)

    # ───────────────────── vLLM call (stream) ─────────────────────
    with st.chat_message("assistant"):
        placeholder = st.empty()
        assistant_reply = ""
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that can see images when provided."},
                    *st.session_state.messages,
                ],
                stream=True,
            )
            for chunk in response:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    assistant_reply += delta.content
                    placeholder.markdown(assistant_reply + "▌")
            placeholder.markdown(assistant_reply)
        except Exception as e:
            assistant_reply = f"❌ Error: {e}"
            placeholder.error(assistant_reply)

    st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
