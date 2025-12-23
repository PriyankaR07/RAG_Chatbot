# rag.py
import os
import re
import sqlite3
import threading
from collections import defaultdict
import numpy as np

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.documents import Document

from ingest import run_ingestion_if_needed
from sql_agent import SQLAgent
from db_setup import create_database_from_sql_files, DB_URI

# ---------- Configuration ----------
EMBEDDING_MODEL = "embeddinggemma:latest"
CHROMA_DB_DIR = "chroma_db"
SCORE_THRESHOLD = 0.1
MAX_PER_SOURCE = 6
CANDIDATE_POOL = 20
FINAL_K = 6
LLM_MODEL = "gemma3:1b"
LLM_TEMPERATURE = 0.1
SQL_DB_URI = DB_URI
# -----------------------------------

class RAGPipeline:
    def __init__(self):
        # 1. SETUP: Create the live SQL database schema from .sql files
        print("[INIT] Setting up SQL database...")
        self.live_db_uri = create_database_from_sql_files(SQL_DB_URI)
        
        # Store DB file path for thread-safe connections
        if self.live_db_uri and self.live_db_uri.startswith("sqlite:///"):
            self.db_file_path = self.live_db_uri.replace("sqlite:///", "")
        else:
            self.db_file_path = "agent_data.db"

        # 2. RAG INGESTION: Ensure ALL documents are ingested
        print("[INIT] Running document ingestion...")
        try:
            # Force re-ingestion to ensure documents are loaded
            if os.path.exists("chroma_db"):
                print("[INIT] Chroma DB exists, checking if empty...")
                # Check if chroma is empty
                try:
                    test_db = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=OllamaEmbeddings(model=EMBEDDING_MODEL))
                    test_collection = test_db.get()
                    if len(test_collection.get('documents', [])) == 0:
                        print("[INIT] Chroma DB is empty, forcing re-ingestion...")
                        # Delete and re-run
                        import shutil
                        shutil.rmtree("chroma_db", ignore_errors=True)
                        run_ingestion_if_needed()
                    else:
                        print(f"[INIT] Chroma DB has {len(test_collection['documents'])} documents")
                except:
                    print("[INIT] Could not check Chroma, forcing ingestion...")
                    run_ingestion_if_needed()
            else:
                print("[INIT] Chroma DB not found, running ingestion...")
                run_ingestion_if_needed()
        except Exception as e:
            print(f"[INIT] Error during ingestion: {e}")
            # Try to run ingestion anyway
            run_ingestion_if_needed()
        
        print("[INIT] ✓ Document ingestion complete")

        # RAG Components
        self.embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

        # 3. Load the Chroma DB with ALL documents
        print("[INIT] Loading persisted Chroma DB...")
        try:
            self.db = Chroma(
                persist_directory=CHROMA_DB_DIR,
                embedding_function=self.embeddings
            )
            
            # Check what's in Chroma
            collection = self.db.get()
            doc_count = len(collection.get('documents', []))
            print(f"[INIT] ✓ Chroma DB loaded with {doc_count} document chunks")
            
            if doc_count > 0:
                # List unique sources
                sources = set()
                for metadata in collection.get('metadatas', []):
                    if metadata and 'source' in metadata:
                        sources.add(os.path.basename(metadata['source']))
                print(f"[INIT] Found documents from {len(sources)} files: {list(sources)[:5]}")
            else:
                print("[INIT] ⚠ WARNING: Chroma DB is empty! No documents were ingested.")
                print("[INIT] Check that you have files in the 'docs' folder.")
                
        except Exception as e:
            print(f"[INIT] ❌ Failed to load Chroma DB: {e}")
            self.db = None

        # LLM for RAG
        self.llm = ChatOllama(model=LLM_MODEL, temperature=LLM_TEMPERATURE)

        # 4. SQL Agent Component - for DIRECT database queries only
        print(f"[INIT] Checking for SQL Agent...")
        try:
            # Check if database file exists
            if os.path.exists(self.db_file_path):
                print(f"[INIT] Database file exists: {self.db_file_path}")
                self.sql_agent = SQLAgent(
                    db_uri=self.live_db_uri,
                    llm_model=LLM_MODEL,
                    llm_temperature=LLM_TEMPERATURE
                )
                print("[INIT] ✓ SQL Agent ready for database queries")
            else:
                print(f"[INIT] Database file not found: {self.db_file_path}")
                self.sql_agent = None
        except Exception as e:
            print(f"[INIT] ⚠ SQL Agent initialization failed: {e}")
            self.sql_agent = None

        self.chat_history = []
        
        # Thread-safe SQL connection for simple queries
        self._thread_local = threading.local()

    def _get_thread_safe_connection(self):
        """Get thread-safe SQLite connection"""
        if not hasattr(self._thread_local, "connection"):
            try:
                self._thread_local.connection = sqlite3.connect(self.db_file_path, check_same_thread=False)
                self._thread_local.connection.row_factory = sqlite3.Row
            except Exception as e:
                print(f"[RAG] Failed to create thread-safe connection: {e}")
                return None
        return self._thread_local.connection

    # --- RAG Helper Methods ---
    @staticmethod
    def _cosine(v1, v2):
        v1 = np.array(v1, dtype=float)
        v2 = np.array(v2, dtype=float)
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 == 0 or n2 == 0: return 0.0
        return float(np.dot(v1, v2) / (n1 * n2))

    @staticmethod
    def _compact_text(text, max_chars=1500):
        if not text: return ""
        t = re.sub(r'\s+', ' ', text).strip()
        return t[:max_chars]

    @staticmethod
    def _dedupe_sentences(text, max_output_chars=1200):
        if not text: return ""
        parts = re.split(r'(?<=[\.\?\!])\s+', text.strip())
        seen = set()
        out = []
        for p in parts:
            s = p.strip()
            if not s: continue
            key = s.lower()
            if key in seen: continue
            seen.add(key)
            out.append(s)
            if sum(len(x) for x in out) > max_output_chars: break
        return " ".join(out)

    def retrieve(self, query, k=FINAL_K, debug=True, force_source=None, source_boost=None,
                 score_threshold=SCORE_THRESHOLD, max_per_source=MAX_PER_SOURCE,
                 candidate_pool=CANDIDATE_POOL):

        if debug: print(f"[retrieve] Query={query!r} k={k}")

        # Check if db is available
        if not hasattr(self, 'db') or self.db is None:
            print("[retrieve] ⚠ Chroma DB not available")
            return []

        # 1. Embed the query
        q_emb = self.embeddings.embed_query(query)

        # 2. Fetch ALL documents with embeddings
        candidates = []
        try:
            if debug: print(f"[retrieve] Fetching documents from Chroma...")
            
            res = self.db.get(include=["documents", "embeddings", "metadatas"])
            docs_list = res.get("documents", [])
            embs_list = res.get("embeddings", [])
            metas_list = res.get("metadatas", [])

            for doc_content, emb_data, meta_data in zip(docs_list, embs_list, metas_list):
                class _D: pass
                d = _D()
                d.page_content = doc_content
                d.metadata = meta_data if meta_data else {}
                d._embedding = emb_data
                candidates.append(d)

        except Exception as e:
            print(f"[retrieve] Error fetching documents: {e}")
            return []

        # 3. Score candidates
        texts, embeddings, metas, extracted = [], [], [], 0
        for c in candidates:
            emb = getattr(c, "_embedding", None)
            if emb is not None:
                embeddings.append(emb)
                texts.append(getattr(c, "page_content", "") or "")
                metas.append(getattr(c, "metadata", {}) or {})
                extracted += 1

        if extracted == 0:
            if debug: print("[retrieve] No documents available in Chroma.")
            return []

        scored = []
        for idx in range(len(texts)):
            meta = metas[idx] if metas else {}
            src = meta.get("source", "unknown") if isinstance(meta, dict) else "unknown"

            if force_source and (force_source.lower() not in src.lower()): 
                continue

            base_score = self._cosine(q_emb, embeddings[idx])
            boost = 1.0
            if source_boost and isinstance(source_boost, dict):
                for key, factor in source_boost.items():
                    if key.lower() in src.lower():
                        try: boost = float(factor)
                        except: boost = 1.0
                        break
            final_score = base_score * boost
            scored.append({"score": final_score, "text": texts[idx], "meta": meta, "src": src})

        scored.sort(key=lambda x: x["score"], reverse=True)
        filtered = [s for s in scored if s["score"] >= score_threshold]

        if debug: 
            print(f"[retrieve] {len(scored)} documents scored, {len(filtered)} above threshold")

        by_source_count = defaultdict(int)
        final_items = []
        for item in filtered:
            if by_source_count[item["src"]] >= max_per_source: continue
            by_source_count[item["src"]] += 1
            final_items.append(item)
            if len(final_items) >= k: break

        if debug and final_items:
            print("[retrieve] Top document matches:")
            for i, it in enumerate(final_items[:3], 1):
                preview = (it["text"] or "")[:150].replace("\n", " ")
                source_name = os.path.basename(it["src"]) if it["src"] != "unknown" else "unknown"
                print(f"  {i}. score={it['score']:.3f} [File: {source_name}] {preview}...")

        docs_out = [Document(page_content=it["text"], metadata=it["meta"]) for it in final_items]
        return docs_out

    def format_context(self, docs):
        if not docs: return ""
        grouped = defaultdict(list)
        for d in docs:
            src = d.metadata.get("source", "unknown") if isinstance(d.metadata, dict) else "unknown"
            # Extract just the filename for cleaner display
            filename = os.path.basename(src) if src != "unknown" else "unknown"
            compacted = self._compact_text(d.page_content, max_chars=2000)
            grouped[filename].append(compacted)

        sections = []
        for filename, pieces in grouped.items():
            merged = " ".join(pieces)
            deduped = self._dedupe_sentences(merged)
            sections.append(f"[From: {filename}]\n{deduped}\n")
        return "\n\n".join(sections)

    def _should_use_sql(self, query: str) -> bool:
        """ONLY use SQL for VERY EXPLICIT database queries"""
        if not self.sql_agent:
            return False
            
        # These keywords indicate user wants to QUERY THE DATABASE directly
        explicit_sql_keywords = [
            "run sql", "execute sql", "sql query", "database query",
            "select from", "query the database", "query database",
            "sql statement", "show tables", "database schema",
            "table structure", "describe table", "what tables are",
            "list tables", "database structure", "tables in database",
            "what are the tables", "show all tables", "list all tables",
            "database tables", "tables available", "available tables",
            "show database", "database schema", "schema of database"
        ]
        
        # Check for very explicit SQL/database keywords
        q = query.lower()
        if any(keyword in q for keyword in explicit_sql_keywords):
            print(f"[Router] SQL trigger: Found explicit database keyword")
            return True
        
        # Only use SQL if explicitly asking about database structure
        if ("database" in q or "sql" in q) and \
           ("what" in q or "how" in q or "show" in q or "list" in q):
            # Check if it's about structure, not content
            structure_words = ["structure", "schema", "tables", "columns", "design"]
            if any(word in q for word in structure_words):
                print(f"[Router] SQL trigger: Database structure query")
                return True
        
        # Check if it's asking to run a specific SQL query
        if q.startswith(("select ", "insert ", "update ", "delete ", "create ")):
            print(f"[Router] SQL trigger: Looks like a direct SQL command")
            return True
        
        return False

    def _simple_database_info(self):
        """Get simple database info using thread-safe connection"""
        conn = self._get_thread_safe_connection()
        if not conn:
            return "Cannot connect to database"
        
        try:
            cursor = conn.cursor()
            
            # Get all tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
            tables = [row[0] for row in cursor.fetchall()]
            
            if not tables:
                return "Database has no tables"
            
            info = f"Database has {len(tables)} table(s):\n\n"
            
            for table in tables:
                # Get column info
                cursor.execute(f"PRAGMA table_info({table})")
                columns = cursor.fetchall()
                
                col_names = [col[1] for col in columns[:5]]  # First 5 columns
                info += f"**Table: {table}**\n"
                info += f"Columns: {', '.join(col_names)}"
                if len(columns) > 5:
                    info += f" (+{len(columns)-5} more)"
                info += "\n\n"
            
            return info
            
        except Exception as e:
            return f"Error reading database: {str(e)}"

    def ask(self, query):
        """Main query handler - ALWAYS check documents first, SQL only for explicit DB queries"""
        
        print(f"\n{'='*60}")
        print(f"[Query] User asked: {query}")
        print(f"{'='*60}")
        
        # FIRST: Always try to find answer in documents
        print("[Router] → Step 1: Searching ALL uploaded documents...")
        docs = []
        
        if hasattr(self, 'db') and self.db is not None:
            docs = self.retrieve(query, k=FINAL_K, debug=True)
        
        if docs:
            print(f"[Router] ✓ Found {len(docs)} relevant document chunks")
            
            # Format context from documents
            context = self.format_context(docs)
            
            if len(context.strip()) > 100:  # Need sufficient context
                print("[Router] ✓ Sufficient context found, generating answer from documents...")
                
                # Use LLM to answer from documents
                messages = [
                    {"role": "system", "content": 
                        "You are a helpful assistant answering questions from uploaded documents. "
                        "Use ONLY the provided context to answer the question. "
                        "The context comes from various files (PDF, CSV, SQL, TXT, etc.). "
                        "When providing information, mention which file it came from using [filename]. "
                        "Be detailed, comprehensive, and accurate. "
                        "If you cannot find the answer in the context, say so clearly. "
                        "Do not make up information or use outside knowledge."},
                    {"role": "user", "content": f"Here are excerpts from uploaded documents:\n\n{context}\n\nQuestion: {query}\n\nPlease answer based ONLY on the provided documents."}
                ]
                
                try:
                    response = self.llm.invoke(messages)
                    answer = response.content
                    
                    # Clean answer
                    lines = [ln.strip() for ln in answer.splitlines() if ln.strip()]
                    unique_lines = []
                    seen = set()
                    for ln in lines:
                        if ln.lower() not in seen:
                            seen.add(ln.lower())
                            unique_lines.append(ln)
                    final_answer = "\n".join(unique_lines)
                    
                    print("[Router] ✓ Successfully generated answer from documents")
                    self.chat_history.append({"user": query, "assistant": final_answer})
                    return final_answer, docs
                    
                except Exception as e:
                    print(f"[Router] ❌ LLM error: {e}")
            else:
                print(f"[Router] ⚠ Context too short ({len(context)} chars), continuing...")
        else:
            print("[Router] ⚠ No relevant documents found or Chroma DB not available")
        
        # SECOND: Check if this is an EXPLICIT SQL/database query
        print("[Router] → Step 2: Checking if this is a database query...")
        should_use_sql = self._should_use_sql(query)
        
        if should_use_sql:
            print("[Router] → This appears to be an explicit database query")
            
            # First try to give simple database info
            simple_info = self._simple_database_info()
            print("[Router] → Providing simple database information")
            
            answer = f"**Database Information:**\n\n{simple_info}\n\n"
            answer += "If you need more specific database queries, try:\n"
            answer += "- 'Show all tables in database'\n"
            answer += "- 'Describe [table_name] table'\n"
            answer += "- 'What columns are in [table_name]?'\n"
            answer += "- 'Show sample data from [table_name]'"
            
            self.chat_history.append({"user": query, "assistant": answer})
            return answer, []
        
        # THIRD: No relevant documents and not a SQL query
        print("[Router] → Step 3: No answer found in documents or SQL")
        
        if not docs and (not hasattr(self, 'db') or self.db is None):
            answer = "❌ **No documents loaded!**\n\n"
            answer += "I couldn't find any documents in the system. Please ensure:\n\n"
            answer += "1. You have files (PDF, CSV, TXT, SQL, etc.) in the 'docs' folder\n"
            answer += "2. The documents have been properly ingested\n"
            answer += "3. Try restarting the application\n\n"
            answer += "**Database Status:**\n"
            if self.sql_agent:
                answer += "✓ Database is available for queries about tables and schemas\n"
            else:
                answer += "⚠ Database not available\n"
            
            return answer, []
        elif not docs:
            answer = "❌ **No matching documents found**\n\n"
            answer += "I searched through your documents but couldn't find information to answer your question.\n\n"
            answer += "**Try:**\n"
            answer += "1. Asking about specific content in your documents\n"
            answer += "2. Referring to file names in your question\n"
            answer += "3. Asking about database structure: 'What tables are in the database?'\n\n"
            
            if self.sql_agent:
                answer += "**Available database tables:**\n"
                simple_info = self._simple_database_info()
                answer += simple_info
            
            return answer, []
        else:
            # We have docs but they weren't relevant enough
            answer = "❌ **Information not found**\n\n"
            answer += "I found some documents but couldn't extract a clear answer to your question.\n\n"
            answer += "**Try asking more specifically:**\n"
            answer += "1. Mention file names in your question\n"
            answer += "2. Ask about specific database tables\n"
            answer += "3. Or ask: 'What files do you have?'\n"
            
            return answer, docs