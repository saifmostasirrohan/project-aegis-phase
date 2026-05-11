import streamlit as st
import requests

# --- 1. UI SETUP ---
st.set_page_config(page_title="Project Aegis", page_icon="🛡️", layout="centered")
st.title("🛡️ Project Aegis Intelligence")
st.caption("Advanced RAG: Hybrid Search (BM25 + Vector) with Cross-Encoder Re-ranking")

# --- 2. SESSION STATE (MEMORY) ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- 3. CHAT INTERFACE ---
user_input = st.chat_input("Ask a question about the research papers...")

if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
        
    # Format chat history for the backend
    history_str = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[:-1]])
    
    # Send to Backend
    with st.chat_message("assistant"):
        with st.spinner("Analyzing documents via Hybrid Search..."):
            try:
                # Call the FastAPI backend
                res = requests.post(
                    "http://127.0.0.1:8001/query", 
                    json={"question": user_input, "chat_history": history_str}
                )
                res.raise_for_status()
                data = res.json()
                answer = data["answer"]
                
                st.markdown(answer)
                st.caption(f"Sources utilized: {data['chunks_used']} verified chunks")
                
                # Save to memory
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                st.error(f"Backend connection failed. Is the FastAPI server running? Error: {e}")
