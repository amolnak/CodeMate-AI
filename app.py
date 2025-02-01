import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    ChatPromptTemplate
)

# 🎨 --- Apply Light Theme with ChatGPT-Style UI ---
def apply_styling():
    st.markdown("""
    <style>
        /* General Background & Font */
        body { background-color: #f8f9fa; color: #333; font-family: Arial, sans-serif; }
        .main { background-color: #ffffff; padding-bottom: 60px; }
        .sidebar .sidebar-content { background-color: #f1f3f4; }
        
        /* Chat Layout */
        .chat-container { display: flex; flex-direction: column; gap: 10px; padding-bottom: 60px; }
        .chat-bubble { padding: 12px 16px; margin: 5px 0; border-radius: 16px; max-width: 75%; }
        .user-message { background-color: #007bff; color: white; align-self: flex-end; text-align: right; }
        .ai-message { background-color: #e9ecef; color: #333; align-self: flex-start; }
        
        /* Sticky Input Box */
        .stChatInput { position: fixed; bottom: 0; width: 100%; background: #ffffff; padding: 10px; border-top: 1px solid #ddd; }
        
        /* Loading Spinner */
        .stSpinner { color: #007bff !important; }

        /* Model Selector */
        .stRadio div[role="radiogroup"] { display: flex; flex-direction: column; gap: 5px; }
        .stRadio div[role="radiogroup"] label { font-weight: bold; }

    </style>
    """, unsafe_allow_html=True)

# 🔹 --- Sidebar Configuration ---
def configure_sidebar():
    with st.sidebar:
        st.header("⚙️ AI Settings")
        selected_model = st.radio(
            "🤖 Select AI Model",
            ["deepseek-r1:1.5b", "deepseek-r1:7b", "qwen2.5:1.5b"],
            index=0
        )
        st.markdown("---")
        st.subheader("📌 Features")
        st.markdown("""
        - 💡 **AI-Powered Coding Help**  
        - 🐞 **Debugging Assistant**  
        - 📝 **Code Explanation & Docs**  
        - ⚡ **Performance Optimization**  
        """)
        st.markdown("---")
        st.markdown("🔗 Powered by [Ollama](https://ollama.ai/) & [LangChain](https://python.langchain.com/)")
    return selected_model

# 🔹 --- AI Engine Initialization ---
def load_ai_model(model):
    return ChatOllama(model=model.lower().replace(" ", ""), base_url="http://localhost:11434", temperature=0.3)

# 🔹 --- System Prompt Setup ---
def system_instruction():
    return SystemMessagePromptTemplate.from_template(
        "You are CodeMate AI, a professional coding assistant. "
        "Provide accurate, optimized, and structured coding advice in a friendly and clear way."
    )

# 🔹 --- Initialize Chat Session ---
def setup_chat_session():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {"role": "ai", "content": "👋 Hello! I'm CodeMate AI. How can I help with your code today?"}
        ]

# 🔹 --- Display Chat Messages (ChatGPT Style) ---
def show_chat():
    chat_area = st.container()
    with chat_area:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for msg in st.session_state.chat_history:
            role_class = "user-message" if msg["role"] == "user" else "ai-message"
            st.markdown(f'<div class="chat-bubble {role_class}">{msg["content"]}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# 🔹 --- Build AI Prompt with History ---
def create_chat_prompt():
    messages = [system_instruction()]
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            messages.append(HumanMessagePromptTemplate.from_template(msg["content"]))
        elif msg["role"] == "ai":
            messages.append(AIMessagePromptTemplate.from_template(msg["content"]))
    return ChatPromptTemplate.from_messages(messages)

# 🔹 --- AI Response Processing ---
def get_ai_reply(prompt, ai_engine):
    pipeline = prompt | ai_engine | StrOutputParser()
    return pipeline.invoke({})

# 🔹 --- Handle User Input ---
def process_chat(user_message, ai_engine):
    if user_message:
        # Add User Message
        st.session_state.chat_history.append({"role": "user", "content": user_message})

        # Show typing indicator
        with st.spinner("🤖 AI is thinking..."):
            chat_prompt = create_chat_prompt()
            ai_reply = get_ai_reply(chat_prompt, ai_engine)

        # Add AI Response
        st.session_state.chat_history.append({"role": "ai", "content": ai_reply})

        # Refresh UI
        st.rerun()

# 🔹 --- Main Function ---
def main():
    apply_styling()
    st.title("💡 CodeMate AI")
    st.caption("🚀 Your AI-Powered Coding Assistant")

    selected_model = configure_sidebar()
    ai_engine = load_ai_model(selected_model)

    setup_chat_session()
    show_chat()

    # Sticky input at bottom (ChatGPT-like input box)
    user_message = st.chat_input("💬 Type a coding question...")
    process_chat(user_message, ai_engine)

# 🔹 --- Run App ---
if __name__ == "__main__":
    main()
