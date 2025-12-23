# newapp.py
import streamlit as st
from rag import RAGPipeline 

st.set_page_config(page_title="Local RAG Chat", layout="centered")
st.title("Offline RAG")

# --- Pipeline Initialization ---
@st.cache_resource
def initialize_pipeline():
    return RAGPipeline()

pipeline = initialize_pipeline()

# --- Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []
    
# Initial greeting message
if not st.session_state.messages:
    sql_available = "✓ SQL Agent available for database queries" if pipeline.sql_agent else "⚠ SQL Agent not available"
    doc_count = len(pipeline.db.get()['documents']) if hasattr(pipeline, 'db') else 0
    doc_status = f"✓ {doc_count} document chunks loaded" if doc_count > 0 else "⚠ No documents loaded"
    
    st.session_state.messages.append({
        "role": "assistant", 
        "content": f"""Hello! I'm your hybrid RAG assistant.

**Status:**
- {doc_status}
- {sql_available}

You can ask me about:
1. Content of your uploaded documents (PDF, CSV, TXT, SQL files)
2. Database queries (tables, schemas, data)
3. Or ask anything about your files!"""
    })

# --- Message Rendering ---
def render_message(role, content):
    avatar = "👤" if role == "user" else "✨"
    with st.chat_message(role, avatar=avatar):
        st.markdown(content) 

for msg in st.session_state.messages:
    render_message(msg["role"], msg["content"])

# --- Chat Input and Response ---
query = st.chat_input("Ask a question...")

if query:
    # 1. Display user query
    st.session_state.messages.append({"role": "user", "content": query})
    render_message("user", query)

    # 2. Get RAG/SQL answer
    with st.spinner("Thinking..."):
        answer, docs = pipeline.ask(query)

    final_answer = answer 

    # 3. Display assistant response
    st.session_state.messages.append({"role": "assistant", "content": final_answer})
    render_message("assistant", final_answer)