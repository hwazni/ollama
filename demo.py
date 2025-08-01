import os
import base64
import streamlit as st
from openai import OpenAI

"""
Image‑chat Streamlit demo for **vLLM** (OpenAI‑compatible) servers.
Compatible with the **OpenAI Python ≥ 1.0** client.

• Upload an image **once**. The app sends it exactly once; afterwards you can
  chat freely while the model remembers the picture.
• When you upload a **different** image or hit **Reset Chat**, context resets.

```bash
pip install vllm[pillow] streamlit openai>=1
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
        help="Base URL where your vLLM OpenAI‑compatible server is running.",
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

    # OpenAI client (>=1.0)
    client = OpenAI(
        api_key=api_key or "EMPTY",  # vLLM ignores it, client requires non‑empty
        base_url=server_url.rstrip("/"),
    )

    st.markdown(
        "If the app fails to connect, ensure that your vLLM server was started "
        "with `--serve-chat-completions` and that **image input** is enabled.")

# ─────────────────────── Utility: image ➜ data URL ───────────────────────

def image_to_data_url(file):
    mime_type = file.type
    encoded = base64.b64encode(file.getvalue()).decode()
    return f"data:{mime_type};base64,{encoded}"

# ──────────────────────── Session‑state bootstrapping ────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []          # Chat history
if "image_data_url" not in st.session_state:
    st.session_state.image_data_url = None  # Current image (base64 URL)
if "image_sent" not in st.session_state:
    st.session_state.image_sent = False     # Has current image been sent?

# ────────────────────────────── Main UI ──────────────────────────────
uploaded_file = st.file_uploader(
    "Upload an image (PNG/JPG/WebP)",
    type=["png", "jpg", "jpeg", "webp"],
)

# Handle new uploads ONLY when the selected file changes --------------
if uploaded_file:
    new_data_url = image_to_data_url(uploaded_file)
    if new_data_url != st.session_state.image_data_url:  # New or changed file
        st.session_state.image_data_url = new_data_url
        st.session_state.image_sent = False              # Force re‑send once
        st.session_state.messages.clear()                # Fresh context
    st.image(uploaded_file, use_column_width=True)

if st.sidebar.button("🔄 Reset Chat"):
    st.session_state.messages.clear()
    st.session_state.image_sent = False

# ───────────────────────── Render existing chat ─────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            if isinstance(msg["content"], list):
                text_parts = [c.get("text", "") for c in msg["content"] if c["type"] == "text"]
                st.markdown(text_parts[0] if text_parts else "")
            else:
                st.markdown(msg["content"])
        else:
            st.markdown(msg["content"])

# ────────────────────────────── Chat input ──────────────────────────────
user_prompt = st.chat_input("Ask something… (image optional after first)")

if user_prompt:
    if not st.session_state.image_data_url and not st.session_state.messages:
        st.warning("Please upload an image first.")
        st.stop()

    # Include image exactly once (first turn after upload) ---------------
    if not st.session_state.image_sent and st.session_state.image_data_url:
        user_msg_content = [
            {"type": "image_url", "image_url": {"url": st.session_state.image_data_url}},
            {"type": "text", "text": user_prompt},
        ]
        st.session_state.image_sent = True
    else:
        user_msg_content = [{"type": "text", "text": user_prompt}]

    st.session_state.messages.append({"role": "user", "content": user_msg_content})

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
