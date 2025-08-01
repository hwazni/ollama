import os
import base64
import streamlit as st
from openai import OpenAI

"""
Streamlit **multimodal chatbot** for vLLM (OpenAI‑compatible) servers
====================================================================

• Works like a *normal* text chatbot.
• If you *also* upload an image, the next message you send will include that
  image so the model can reason about it. There’s **no need** to re‑upload the
  image for follow‑up questions.
• Upload a *different* image at any point to pivot the conversation; the new
  picture will be attached to your very next message only.

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

    server_url = st.text_input(
        "vLLM Server URL",
        value=os.getenv("VLLM_SERVER_URL", "http://localhost:8000/v1"),
        help="Where your vLLM OpenAI‑compatible endpoint lives.",
    )

    api_key = st.text_input(
        "API Key (optional)", type="password", value=os.getenv("OPENAI_API_KEY", "")
    )

    model = st.text_input(
        "Model name on server", value=os.getenv("VLLM_MODEL", "liuhaotian/llava-v1.6-34b")
    )

    if not server_url:
        st.error("Please enter the server URL, e.g. http://localhost:8000/v1")
        st.stop()

    client = OpenAI(api_key=api_key or "EMPTY", base_url=server_url.rstrip("/"))

    st.markdown("Upload an image *anytime*; it’ll be attached to your **next** message only.")

# ─────────────────────── Helper: image ➜ base64 URL ───────────────────────

def image_to_data_url(file):
    mime_type = file.type
    encoded = base64.b64encode(file.getvalue()).decode()
    return f"data:{mime_type};base64,{encoded}"

# ───────────────────── Session‑state init ─────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []      # Conversation history
if "pending_image" not in st.session_state:
    st.session_state.pending_image = None  # Data URL waiting to be sent

# ───────────────────────── Image upload ─────────────────────────
uploaded_file = st.file_uploader("Optional image (PNG/JPG/WebP)", type=["png", "jpg", "jpeg", "webp"])
if uploaded_file:
    data_url = image_to_data_url(uploaded_file)
    # Store/overwrite pending image; show preview
    st.session_state.pending_image = data_url
    st.image(uploaded_file, use_column_width=True)

# ───────────────────────── Chat history render ─────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user" and isinstance(msg["content"], list):
            # Extract text portion
            text = next((c["text"] for c in msg["content"] if c["type"] == "text"), "")
            st.markdown(text)
        else:
            st.markdown(msg["content"])

# ───────────────────────── Chat input ─────────────────────────
user_text = st.chat_input("Message…")
if user_text:
    # Build user content; attach image if there’s a pending one
    if st.session_state.pending_image:
        user_content = [
            {"type": "image_url", "image_url": {"url": st.session_state.pending_image}},
            {"type": "text", "text": user_text},
        ]
        st.session_state.pending_image = None  # Consume it
    else:
        user_content = [{"type": "text", "text": user_text}]

    st.session_state.messages.append({"role": "user", "content": user_content})

    with st.chat_message("user"):
        st.markdown(user_text)

    # ─────────────────────── vLLM call (stream) ───────────────────────
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
