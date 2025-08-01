import os
import base64
import streamlit as st
from openai import OpenAI

"""
Image-chat Streamlit demo for **vLLM** (OpenAI-compatible) servers.
Compatible with the **OpenAI Python ≥ 1.0** client.
Launch a server (example below), point the sidebar at it, upload an image, and
start chatting. Responses stream in real time.

```bash
pip install vllm[pillow]
python -m vllm.entrypoints.openai.api_server \
    --model liuhaotian/llava-v1.6-34b \
    --image-input-size 576
```
"""

st.set_page_config(page_title="Image Chat · vLLM", layout="centered")
st.title("🖼️ Chat with Your Image · vLLM")

# ───────────────────────────── Sidebar ──────────────────────────────
with st.sidebar:
    st.header("🔌 Connection Settings")

    server_url = st.text_input(
        "vLLM Server URL",
        value=os.getenv("VLLM_SERVER_URL", "http://localhost:8000/v1"),
        help="Base URL where your vLLM OpenAI-compatible server is running.",
    )

    api_key = st.text_input(
        "API Key (optional)",
        type="password",
        value=os.getenv("OPENAI_API_KEY", ""),
        help="Leave blank for local vLLM; a dummy key will be supplied.",
    )

    model = st.text_input(
        "Model name on server",
        value=os.getenv("VLLM_MODEL", "liuhaotian/llava-v1.6-34b"),
        help="Must match the --model flag used when launching vLLM.",
    )

    if not server_url:
        st.error("Enter your vLLM server URL (e.g. http://localhost:8000/v1)")
        st.stop()

    # Create a dedicated OpenAI client instance (>=1.0 style)
    client = OpenAI(
        api_key=api_key or "EMPTY",  # vLLM ignores it, client requires non-empty
        base_url=server_url.rstrip("/"),
    )

    st.markdown(
        "If the app fails to connect, ensure that your vLLM server was started "
        "with the **--serve-chat-completions** flag (added by default since v0.4) "
        "and that **image input** is enabled for your chosen model.")

# ─────────────────────── Utility: image ➜ data URL ───────────────────────

def image_to_data_url(uploaded_file):
    """Convert a Streamlit-uploaded file into a base64 data URL."""
    mime_type = uploaded_file.type
    encoded = base64.b64encode(uploaded_file.getvalue()).decode()
    return f"data:{mime_type};base64,{encoded}"

# ──────────────────────── Session-state bootstrapping ────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []  # chat history for OpenAI format
if "image_data_url" not in st.session_state:
    st.session_state.image_data_url = None

# ────────────────────────────── Main UI ──────────────────────────────
uploaded_file = st.file_uploader(
    "Upload an image (PNG/JPG/WebP)",
    type=["png", "jpg", "jpeg", "webp"],
)

if uploaded_file:
    st.image(uploaded_file, use_column_width=True)
    st.session_state.image_data_url = image_to_data_url(uploaded_file)

if st.sidebar.button("🔄 Reset Chat"):
    st.session_state.messages.clear()

# ───────────────────────── Render existing chat ─────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            # Extract the textual part from the multimodal payload
            text_snippets = [c.get("text", "") for c in msg["content"] if c["type"] == "text"]
            st.markdown(text_snippets[0] if text_snippets else "")
        else:
            st.markdown(msg["content"])

# ────────────────────────────── Chat input ──────────────────────────────
user_prompt = st.chat_input("Ask something about the image…")

if user_prompt:
    if not st.session_state.image_data_url:
        st.warning("Please upload an image first.")
        st.stop()

    # Build multimodal user message
    user_msg = [
        {"type": "image_url", "image_url": {"url": st.session_state.image_data_url}},
        {"type": "text", "text": user_prompt},
    ]

    st.session_state.messages.append({"role": "user", "content": user_msg})

    with st.chat_message("user"):
        st.markdown(user_prompt)

    # ─────────────────────── Call vLLM (streaming) ───────────────────────
    with st.chat_message("assistant"):
        placeholder = st.empty()
        assistant_resp = ""

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an AI assistant that can see images and answer questions about them.",
                    },
                    *st.session_state.messages,
                ],
                stream=True,
            )

            for chunk in response:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    assistant_resp += delta.content
                    placeholder.markdown(assistant_resp + "▌")
            placeholder.markdown(assistant_resp)

        except Exception as e:
            assistant_resp = f"❌ Error: {e}"
            placeholder.error(assistant_resp)

    st.session_state.messages.append({"role": "assistant", "content": assistant_resp})
