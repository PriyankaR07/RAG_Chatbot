# newapp.py - WhatsApp-style UI with Chat History
import streamlit as st
from rag import RAGPipeline 
import os
from datetime import datetime
import json

# Page config
st.set_page_config(
    page_title="RAG Chat",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - WhatsApp-inspired design
st.markdown("""
<style>
    /* Main background */
    .main {
        background: linear-gradient(to bottom, #00a884 0%, #00a884 15%, #0b141a 15%, #0b141a 100%);
    }
    
    /* Hide default header */
    header {visibility: hidden;}
    
    /* Sidebar - Chat list style */
    [data-testid="stSidebar"] {
        background-color: #111b21;
        padding-top: 0;
    }
    
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1rem;
    }
    
    /* Chat messages */
    .stChatMessage {
        border-radius: 8px;
        padding: 8px 12px;
        margin: 4px 0;
        max-width: 80%;
    }
    
    /* User message - right side, green */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background-color: #005c4b;
        margin-left: auto;
        margin-right: 0;
    }
    
    /* Assistant message - left side, dark gray */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        background-color: #202c33;
        margin-right: auto;
        margin-left: 0;
    }
    
    /* Hide avatars */
    [data-testid="chatAvatarIcon-user"], 
    [data-testid="chatAvatarIcon-assistant"] {
        display: none;
    }
    
    /* Chat input */
    .stChatInput {
        background-color: #2a3942;
        border-radius: 8px;
        border: none;
    }
    
    .stChatInput textarea {
        background-color: #2a3942;
        color: #e9edef;
        border: none;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #00a884;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 500;
        width: 100%;
    }
    
    .stButton>button:hover {
        background-color: #06cf9c;
    }
    
    /* Text color */
    .stMarkdown, p, label {
        color: #e9edef;
    }
    
    /* Search box */
    .stTextInput input {
        background-color: #202c33;
        color: #e9edef;
        border: none;
        border-radius: 8px;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #202c33;
        color: #e9edef;
        border-radius: 8px;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #00a884;
    }
    
    /* Divider */
    hr {
        border-color: #2a3942;
    }
    
    /* Chat history items */
    .chat-history-item {
        background-color: #202c33;
        padding: 12px;
        border-radius: 8px;
        margin: 8px 0;
        cursor: pointer;
        transition: background-color 0.2s;
    }
    
    .chat-history-item:hover {
        background-color: #2a3942;
    }
    
    .chat-title {
        color: #e9edef;
        font-weight: 500;
        margin-bottom: 4px;
    }
    
    .chat-preview {
        color: #8696a0;
        font-size: 0.85em;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    
    .chat-time {
        color: #8696a0;
        font-size: 0.75em;
        text-align: right;
    }
</style>
""", unsafe_allow_html=True)

# --- Pipeline Initialization ---
@st.cache_resource
def initialize_pipeline():
    return RAGPipeline()

pipeline = initialize_pipeline()

# --- Chat History Management ---
HISTORY_FILE = "chat_history.json"

def load_chat_history():
    """Load all chat sessions from file"""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return []
    return []

def save_chat_history(history):
    """Save chat sessions to file"""
    try:
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
    except Exception as e:
        st.error(f"Error saving history: {e}")

def create_new_chat():
    """Create a new chat session"""
    chat_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    return {
        "id": chat_id,
        "title": "New Chat",
        "timestamp": datetime.now().isoformat(),
        "messages": []
    }

def get_chat_title(messages):
    """Generate title from first user message"""
    for msg in messages:
        if msg["role"] == "user":
            title = msg["content"][:50]
            return title + "..." if len(msg["content"]) > 50 else title
    return "New Chat"

# --- Session State Initialization ---
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = load_chat_history()
    if not st.session_state.chat_sessions:
        st.session_state.chat_sessions = [create_new_chat()]

if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = st.session_state.chat_sessions[0]["id"]

if "messages" not in st.session_state:
    current_chat = next(
        (c for c in st.session_state.chat_sessions if c["id"] == st.session_state.current_chat_id),
        None
    )
    st.session_state.messages = current_chat["messages"] if current_chat else []

# --- Sidebar - Chat History ---
with st.sidebar:
    # Header
    st.markdown("### 💬 RAG Assistant")
    st.markdown("---")
    
    # New Chat Button
    if st.button("➕ New Chat", use_container_width=True):
        new_chat = create_new_chat()
        st.session_state.chat_sessions.insert(0, new_chat)
        st.session_state.current_chat_id = new_chat["id"]
        st.session_state.messages = []
        save_chat_history(st.session_state.chat_sessions)
        st.rerun()
    
    st.markdown("---")
    
    # Search box
    search_query = st.text_input("🔍 Search chats...", label_visibility="collapsed", placeholder="Search chats...")
    
    st.markdown("### 📝 Chat History")
    
    # Display chat history
    filtered_chats = st.session_state.chat_sessions
    if search_query:
        filtered_chats = [
            chat for chat in st.session_state.chat_sessions
            if search_query.lower() in chat.get("title", "").lower()
            or any(search_query.lower() in msg["content"].lower() for msg in chat.get("messages", []))
        ]
    
    if filtered_chats:
        for chat in filtered_chats:
            is_current = chat["id"] == st.session_state.current_chat_id
            
            # Create columns for chat item and delete button
            col1, col2 = st.columns([5, 1])
            
            with col1:
                # Chat button
                button_type = "primary" if is_current else "secondary"
                
                # Get chat preview
                chat_title = chat.get("title", "New Chat")
                chat_messages = chat.get("messages", [])
                preview = ""
                if chat_messages:
                    for msg in reversed(chat_messages):
                        if msg["role"] == "user":
                            preview = msg["content"][:60] + "..."
                            break
                
                # Format timestamp
                try:
                    timestamp = datetime.fromisoformat(chat["timestamp"])
                    time_str = timestamp.strftime("%b %d, %H:%M")
                except:
                    time_str = ""
                
                if st.button(
                    f"**{chat_title}**\n{preview}\n*{time_str}*",
                    key=f"chat_{chat['id']}",
                    use_container_width=True,
                    type=button_type
                ):
                    st.session_state.current_chat_id = chat["id"]
                    st.session_state.messages = chat["messages"]
                    st.rerun()
            
            with col2:
                # Delete button
                if st.button("🗑️", key=f"del_{chat['id']}", help="Delete chat"):
                    st.session_state.chat_sessions = [
                        c for c in st.session_state.chat_sessions if c["id"] != chat["id"]
                    ]
                    if chat["id"] == st.session_state.current_chat_id:
                        if st.session_state.chat_sessions:
                            st.session_state.current_chat_id = st.session_state.chat_sessions[0]["id"]
                            st.session_state.messages = st.session_state.chat_sessions[0]["messages"]
                        else:
                            new_chat = create_new_chat()
                            st.session_state.chat_sessions = [new_chat]
                            st.session_state.current_chat_id = new_chat["id"]
                            st.session_state.messages = []
                    save_chat_history(st.session_state.chat_sessions)
                    st.rerun()
    else:
        st.info("No chats found")
    
    st.markdown("---")
    
    # System info
    with st.expander("📊 System Status"):
        if hasattr(pipeline, 'db') and pipeline.db is not None:
            try:
                collection = pipeline.db.get()
                doc_count = len(collection.get('documents', []))
                st.metric("Documents", f"{doc_count:,}")
            except:
                st.warning("⚠️ No documents")
        else:
            st.warning("⚠️ No documents")
        
        sql_status = "✅ Connected" if pipeline.sql_agent else "❌ Not available"
        st.metric("Database", sql_status)

# --- Main Chat Area ---
# Header
st.markdown("""
<div style='background-color: #202c33; padding: 16px; border-radius: 0; margin: -1rem -1rem 1rem -1rem;'>
    <h2 style='margin: 0; color: #e9edef;'>💬 Chat with Your Documents</h2>
</div>
""", unsafe_allow_html=True)

# Welcome message for new chats
if not st.session_state.messages:
    doc_count = len(pipeline.db.get()['documents']) if hasattr(pipeline, 'db') and pipeline.db else 0
    
    welcome = f"""👋 **Welcome!**

I can help you with:
- 📄 Questions about your documents
- 🗄️ Database queries
- 🔍 Information search

**Status:** {doc_count} document chunks loaded
{"✅ Database connected" if pipeline.sql_agent else "⚠️ Database not available"}

Try asking: *"What files do you have?"* or *"What's in the Excel sheet?"*
"""
    
    st.session_state.messages.append({
        "role": "assistant",
        "content": welcome
    })

# Display messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
query = st.chat_input("Type a message...")

if query:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
    
    # Get response
    with st.spinner("Thinking..."):
        try:
            answer, docs = pipeline.ask(query)
        except Exception as e:
            answer = f"❌ Error: {str(e)}"
            docs = []
    
    # Add assistant message
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
    
    # Update chat history
    current_chat = next(
        (c for c in st.session_state.chat_sessions if c["id"] == st.session_state.current_chat_id),
        None
    )
    if current_chat:
        current_chat["messages"] = st.session_state.messages
        current_chat["title"] = get_chat_title(st.session_state.messages)
        current_chat["timestamp"] = datetime.now().isoformat()
        save_chat_history(st.session_state.chat_sessions)
    
    st.rerun()