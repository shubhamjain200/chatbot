
import streamlit as st
from langchain_groq import ChatGroq
import joblib
import os

st.set_page_config(page_title="My AI Chatbot", page_icon="🤖")

st.title("🤖 My AI Chatbot")

# Initialize LLM with streaming
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    streaming=True
)

# Load Chat History
if "chat" not in st.session_state:

    if os.path.exists("history.pkl"):
        st.session_state.chat = joblib.load("history.pkl")
    else:
        st.session_state.chat = [
            ("system", "you are a helpful assistant")
        ]

# Display Previous Chat
for role, message in st.session_state.chat:

    if role == "system":
        continue

    if role == "ai":
        with st.chat_message("assistant"):
            st.write(message)

    elif role == "user":
        with st.chat_message("user"):
            st.write(message)

# User Input
prompt = st.chat_input("Enter your prompt...")

if prompt:

    # Display User Message
    with st.chat_message("user"):
        st.write(prompt)

    st.session_state.chat.append(("user", prompt))

    # Stream AI Response
    with st.chat_message("assistant"):

        response_placeholder = st.empty()
        full_response = ""

        for chunk in llm.stream(st.session_state.chat):

            if chunk.content:
                full_response += chunk.content
                response_placeholder.markdown(full_response + "▌")

        response_placeholder.markdown(full_response)

    # Save AI Response
    st.session_state.chat.append(("ai", full_response))

    joblib.dump(st.session_state.chat, "history.pkl")
