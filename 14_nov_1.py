
from langchain_groq import ChatGroq
import streamlit as st
import joblib
import os

# Initialize LLM
llm = ChatGroq(
    model="openai/gpt-oss-120b"
)

st.title("🤖 Groq AI Chatbot")

# Load history
if os.path.exists("history.pkl"):
    chat = joblib.load("history.pkl")
else:
    chat = [
        ("system", "You are a helpful assistant")
    ]

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! How can I help you?"
        }
    ]

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User input
prompt = st.chat_input("Enter your prompt...")

if prompt:

    # Display user message
    st.chat_message("user").write(prompt)

    # Add to history
    chat.append(("user", prompt))

    # Convert history to LangChain messages
    response = llm.invoke(chat)

    llm_response = response.content

    # Display AI response
    st.chat_message("assistant").write(llm_response)

    # Update session state
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    st.session_state.messages.append(
        {"role": "assistant", "content": llm_response}
    )

    # Save conversation
    chat.append(("assistant", llm_response))
    joblib.dump(chat, "history.pkl")
