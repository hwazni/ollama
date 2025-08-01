import os
import base64
import streamlit as st
import openai

st.set_page_config(page_title="Image Chat · vLLM", layout="centered")

st.title("🖼️ Chat with Your Image · vLLM")

"""------------------------------------------------------------------------
This Streamlit app lets you talk to an image using any **vLLM** server that
implements the OpenAI-compatible /v1/chat/completions endpoint (e.g. started
via `python -m vllm.entrypoints.openai.api_server`).
------------------------------------------------------------------------"""

# --------------------------- Sidebar: connection ---------------------------
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
        help="Leave blank for local vLLM servers; the library still sends a dummy key.",
    )

    model = st.text_input(
        "Model name on server",
        value=os.getenv("VLLM_MODEL", "liuhaotian/llava-v1.6-34b"),
        help="Must match the `--model` flag you used when launching vLLM.",
    )

    # Apply settings to OpenAI client
    if server_url:
        openai.api_base = server_url.rstrip("/")
    else:
        st.error("Please enter your vLLM server URL (e.g. http://localhost:8000/v1)")
        st.stop()

    openai.api_key = api_key or "EMPTY"  # vLLM ignores the key but the client expects one
    openai.api_type = "openai"            # Explicit for clarity

    st.markdown("""
    **Running vLLM**
    ```bash
    pip install vllm[pillow] # ensures image support
    python -m vllm.entrypoints.openai.api_server \
        --model liuhaotian/llava-v1.6-34b \
        --image-input-size 576           # or as required by your model
    ```
    """, help="Quick start for a local multimodal vLLM server.")

# ------------------------ Utility: image ➜ data URL ------------------------

def image_to_data_url(uploaded_file):
    """Convert a Streamlit‑uploaded file to a base64 data URL usable by OpenAI."""
    mime_type = uploaded_file.type  # e.g. image/png
    encoded = base64.b64encode(uploaded_file.getvalue()).decode()
    return f"data:{mime_type};base64,{encoded}"

# --------------------------- Session state init ---------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # conversation formatted for OpenAI
if "image_data_url" not in st.session_state:
    st.session_state.image_data_url = None

# ------------------------------- UI layout -------------------------------
uploaded_file = st.file_uploader(
    "Upload an image (PNG/JPG/WebP)", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=False
)

if uploaded_file:
    st.image(uploaded_file, use_column_width=True)
    st.session_state.image_data_url = image_to_data_url(uploaded_file)

if st.sidebar.button("🔄 Reset Chat"):
    st.session_state.messages.clear()

# Render chat history
for msg in st.session_state.messages:
    role = msg["role"]
    if role == "assistant":
        with st.chat_message("assistant"):
            st.markdown(msg["content"])
    elif role == "user":
        with st.chat_message("user"):
            # msg["content"] is a list for multimodal user messages
            text_snippets = [c.get("text", "") for c in msg["content"] if c["type"] == "text"]
            if text_snippets:
                st.markdown(text_snippets[0])

# ------------------------------- Chat input -------------------------------
user_prompt = st.chat_input("Ask something about the image…")

if user_prompt:

    if not st.session_state.image_data_url:
        st.warning("Please upload an image first.")
        st.stop()

    # Multimodal user message (image + text)
    user_content = [
        {"type": "image_url", "image_url": {"url": st.session_state.image_data_url}},
        {"type": "text", "text": user_prompt},
    ]

    st.session_state.messages.append({"role": "user", "content": user_content})

    with st.chat_message("user"):
        st.markdown(user_prompt)

    # ------------------------ Call vLLM OpenAI API ------------------------
    with st.chat_message("assistant"):
        placeholder = st.empty()
        assistant_response = ""

        try:
            # Prepend a system prompt for context
            all_messages = [
                {
                    "role": "system",
                    "content": "You are an AI assistant that can see images and answer questions about them.",
                }
            ] + st.session_state.messages

            response = openai.ChatCompletion.create(
                model=model,
                messages=all_messages,
                stream=True,
            )

            for chunk in response:
                delta = chunk.choices[0].delta
                if "content" in delta:
                    assistant_response += delta.content
                    placeholder.markdown(assistant_response + "▌")
            placeholder.markdown(assistant_response)

        except Exception as e:
            assistant_response = f"❌ Error: {e}"
            placeholder.error(assistant_response)

    # Save assistant response
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
