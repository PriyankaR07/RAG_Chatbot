# ingest.py
from pathlib import Path
import os
import re
from collections import defaultdict
import textwrap
import logging

# LangChain / Chroma imports
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    UnstructuredExcelLoader
)
from langchain_community.vectorstores.utils import filter_complex_metadata 
from langchain_unstructured.document_loaders import UnstructuredLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings 
from langchain_chroma import Chroma
from langchain_core.documents import Document
from tempfile import TemporaryDirectory

# OCR Handling
try:
    from pdf2image import convert_from_path
    import pytesseract
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

# --------- Configuration ----------
DOCS_DIR = "docs"
DB_DIR = "chroma_db"
EMBEDDING_MODEL = "embeddinggemma:latest"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 400
# ----------------------------------

def ocr_pdf_to_documents(pdf_path):
    """Convert PDF pages to text via OCR and return a list of LangChain Documents."""
    docs = []
    if not OCR_AVAILABLE:
        print(f"[ingest][ocr] OCR dependencies not available (pdf2image/pytesseract). Skipping OCR for {pdf_path}.")
        return docs

    print(f"[ingest][ocr] Running OCR fallback on {pdf_path} ...")
    try:
        with TemporaryDirectory() as tmpdir:
            images = convert_from_path(pdf_path, dpi=200, output_folder=tmpdir)
            for i, img in enumerate(images):
                try:
                    text = pytesseract.image_to_string(img)
                except Exception as e:
                    print(f"[ingest][ocr] pytesseract failed on page {i}: {e}")
                    text = ""
                if text and text.strip():
                    docs.append(Document(page_content=text, metadata={"source": Path(pdf_path).name}))
    except Exception as e:
        print(f"[ingest][ocr] Failed to OCR {pdf_path}: {e}")
    print(f"[ingest][ocr] OCR produced {len(docs)} page documents for {pdf_path}")
    return docs

def load_all_documents():
    docs = []
    path = Path(DOCS_DIR)

    if not path.exists():
        raise ValueError(f"Docs directory {DOCS_DIR} does not exist. Create it and add documents.")

    for file in sorted(path.iterdir()):
        try:
            suffix = file.suffix.lower()

            # SQL FILES: Load content as document (not just database schema)
            if suffix == ".sql":
                print(f"[ingest] Loading SQL file content → {file}")
                try:
                    # Read SQL file content
                    content = file.read_text(encoding='utf-8', errors='ignore')
                    
                    # Clean up the SQL for readability (replace multiple spaces/newlines with single space)
                    cleaned_content = re.sub(r'\s+', ' ', content).strip()
                    
                    # Extract key information from SQL file
                    tables_found = re.findall(r'CREATE TABLE\s+(\w+)', content, re.IGNORECASE)
                    inserts_found = re.findall(r'INSERT INTO\s+(\w+)', content, re.IGNORECASE)
                    selects_found = re.findall(r'SELECT.*?FROM\s+(\w+)', content, re.IGNORECASE | re.DOTALL)
                    
                    # Create a comprehensive summary of the SQL file
                    summary = f"SQL FILE: {file.name}\n"
                    summary += f"File Size: {len(content)} characters\n"
                    
                    if tables_found:
                        summary += f"📊 Tables Defined: {', '.join(tables_found)}\n"
                    
                    # Also look for column definitions
                    table_details = []
                    for table_match in re.finditer(r'CREATE TABLE\s+(\w+)\s*\(([^)]+)\)', content, re.IGNORECASE | re.DOTALL):
                        table_name = table_match.group(1)
                        columns = table_match.group(2)
                        # Extract column names (simple extraction)
                        col_matches = re.findall(r'\b(\w+)\s+[A-Z]+', columns[:500])
                        if col_matches:
                            table_details.append(f"  • {table_name}: {', '.join(col_matches[:5])}")
                            if len(col_matches) > 5:
                                table_details[-1] += f" (+{len(col_matches)-5} more)"
                    
                    if table_details:
                        summary += "📋 Table Structures:\n" + "\n".join(table_details) + "\n"
                    
                    if inserts_found:
                        unique_inserts = list(set(inserts_found))
                        summary += f"📝 Data Inserts Into: {', '.join(unique_inserts)}\n"
                    
                    # Include first 2500 chars of actual SQL content
                    preview = cleaned_content[:2500]
                    if len(cleaned_content) > 2500:
                        preview += "... [truncated]"
                    
                    full_content = f"{summary}\n\n📄 SQL CONTENT PREVIEW:\n```sql\n{preview}\n```"
                    
                    # Create document from SQL file
                    sql_doc = Document(
                        page_content=full_content,
                        metadata={
                            "source": file.name,
                            "file_type": "sql",
                            "tables": tables_found,
                            "size": len(content),
                            "num_tables": len(tables_found),
                            "num_inserts": len(inserts_found)
                        }
                    )
                    docs.append(sql_doc)
                    print(f"[ingest] ✓ SQL file processed: {file.name} ({len(tables_found)} tables)")
                    
                except Exception as e:
                    print(f"[ingest] ❌ Failed to load SQL file {file}: {e}")
                    # Create a simple document anyway
                    try:
                        simple_doc = Document(
                            page_content=f"SQL File: {file.name}\nUnable to parse content fully.",
                            metadata={"source": file.name, "file_type": "sql"}
                        )
                        docs.append(simple_doc)
                    except:
                        pass
                continue  # Skip the rest of the loop for SQL files

            # PDF FILES
            if suffix == ".pdf":
                print(f"[ingest] Loading PDF → {file}")
                try:
                    pdf_docs = PyPDFLoader(str(file)).load()
                    non_empty_pages = sum(1 for d in pdf_docs if (d.page_content or "").strip())
                    # If PDF is mostly empty images, try OCR
                    if non_empty_pages < max(1, len(pdf_docs) // 2):
                        print("[ingest] PDF text extraction is sparse — attempting OCR fallback.")
                        ocr_docs = ocr_pdf_to_documents(str(file))
                        if ocr_docs:
                            pdf_docs = ocr_docs
                    for d in pdf_docs:
                        d.metadata["source"] = file.name
                    docs.extend(pdf_docs)
                    print(f"[ingest] ✓ PDF loaded: {file.name} ({len(pdf_docs)} pages)")
                except Exception as e:
                    print(f"[ingest] Standard PDF load failed, trying OCR directly: {e}")
                    ocr_docs = ocr_pdf_to_documents(str(file))
                    docs.extend(ocr_docs)

            # TEXT FILES
            elif suffix == ".txt":
                print(f"[ingest] Loading TXT → {file}")
                txt_docs = TextLoader(str(file), encoding='utf-8', autodetect_encoding=True).load()
                for d in txt_docs:
                    d.metadata["source"] = file.name
                docs.extend(txt_docs)
                print(f"[ingest] ✓ TXT loaded: {file.name}")

            # CSV FILES
            elif suffix == ".csv":
                print(f"[ingest] Loading CSV → {file}")
                csv_docs = CSVLoader(str(file)).load()
                for d in csv_docs:
                    d.metadata["source"] = file.name
                docs.extend(csv_docs)
                print(f"[ingest] ✓ CSV loaded: {file.name}")

            # EXCEL FILES
            elif suffix in [".xls", ".xlsx"]:
                print(f"[ingest] Loading XLSX/XLS (Unstructured) → {file}")
                excel_docs = UnstructuredExcelLoader(str(file)).load()
                for d in excel_docs:
                    d.metadata["source"] = file.name
                docs.extend(excel_docs)
                print(f"[ingest] ✓ Excel loaded: {file.name}")

            # OTHER DOCUMENT TYPES (Word, PowerPoint, HTML)
            elif suffix in [".doc", ".docx", ".ppt", ".pptx", ".html", ".htm"]:
                print(f"[ingest] Loading {suffix.upper()} (Unstructured) → {file}")
                other_docs = UnstructuredLoader(str(file)).load()
                for d in other_docs:
                    d.metadata["source"] = file.name
                docs.extend(other_docs)
                print(f"[ingest] ✓ {suffix} loaded: {file.name}")

            # UNSUPPORTED FILES
            else:
                print(f"[ingest] ⚠ Skipping unsupported file type: {file}")

        except Exception as e:
            print(f"[ingest] ❌ Failed to load {file}: {e}")

    # Summary of loaded documents
    if len(docs) == 0:
        print(f"[ingest] ⚠ WARNING: No documents found or extracted in ./{DOCS_DIR} folder!")
    else:
        # Count by file type
        file_types = defaultdict(int)
        for doc in docs:
            source = doc.metadata.get("source", "")
            if source.endswith(".sql"):
                file_types["sql"] += 1
            elif source.endswith(".pdf"):
                file_types["pdf"] += 1
            elif source.endswith(".txt"):
                file_types["txt"] += 1
            elif source.endswith(".csv"):
                file_types["csv"] += 1
            elif source.endswith((".xls", ".xlsx")):
                file_types["excel"] += 1
            else:
                file_types["other"] += 1
        
        print(f"[ingest] ✅ Loaded {len(docs)} raw document sections.")
        print(f"[ingest] File type breakdown: {dict(file_types)}")
        
    return docs

def split_into_chunks(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)
    print(f"[ingest] Created {len(chunks)} chunks.")
    
    # Show chunk size distribution
    if chunks:
        sizes = [len(chunk.page_content) for chunk in chunks[:10]]
        print(f"[ingest] Sample chunk sizes: {sizes}")
    
    return chunks

def build_vectorstore(chunks):
    if not chunks:
        print("[ingest] ❌ No chunks to ingest.")
        return None
        
    print(f"[ingest] Building embeddings using model: {EMBEDDING_MODEL}")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    try:
        vectordb = Chroma.from_documents(
            chunks,
            embeddings,
            persist_directory=DB_DIR
        )
        print("[ingest] ✅ Chroma DB built and persisted.")
        
        # Verify it was created
        collection = vectordb.get()
        print(f"[ingest] Verification: Chroma DB contains {len(collection.get('documents', []))} document chunks.")
        
        return vectordb
    except Exception as e:
        print(f"[ingest] ❌ Failed to build Chroma DB: {e}")
        return None

def run_ingestion_if_needed():
    """Run ingestion only if needed (new files or DB doesn't exist)"""
    
    # Check if docs directory exists
    if not os.path.exists(DOCS_DIR):
        print(f"[ingest] ❌ Docs directory '{DOCS_DIR}' does not exist. Creating it...")
        os.makedirs(DOCS_DIR, exist_ok=True)
        print(f"[ingest] ✅ Created '{DOCS_DIR}'. Please add documents and restart.")
        return

    # Check if there are any files in docs directory
    files = list(Path(DOCS_DIR).iterdir())
    if not files:
        print(f"[ingest] ⚠ No files found in '{DOCS_DIR}'. Please add documents.")
        return
    
    # More robust check for whether to re-ingest
    force_reingest = False
    
    # Check if DB dir doesn't exist
    if not os.path.exists(DB_DIR):
        print(f"[ingest] No DB directory '{DB_DIR}' found → Running ingestion...")
        force_reingest = True
    
    # Check if DB dir is empty
    elif not os.listdir(DB_DIR):
        print(f"[ingest] DB directory '{DB_DIR}' is empty → Running ingestion...")
        force_reingest = True
    
    # Check if there are new files in docs folder
    else:
        try:
            # Get list of current files in docs
            current_files = sorted([f.name for f in Path(DOCS_DIR).iterdir() if f.is_file()])
            
            # Check if we have a record of what was ingested
            record_file = os.path.join(DB_DIR, "ingestion_record.txt")
            if os.path.exists(record_file):
                with open(record_file, 'r', encoding='utf-8') as f:
                    previous_files = f.read().splitlines()
                
                current_set = set(current_files)
                previous_set = set(previous_files)
                
                if current_set != previous_set:
                    print(f"[ingest] 📦 New or changed files detected:")
                    new_files = current_set - previous_set
                    removed_files = previous_set - current_set
                    if new_files:
                        print(f"[ingest]   Added: {', '.join(new_files)}")
                    if removed_files:
                        print(f"[ingest]   Removed: {', '.join(removed_files)}")
                    print(f"[ingest]   → Running ingestion...")
                    force_reingest = True
                else:
                    print(f"[ingest] ✅ DB is up-to-date with current files.")
            else:
                print(f"[ingest] No ingestion record found → Running ingestion...")
                force_reingest = True
                
        except Exception as e:
            print(f"[ingest] ⚠ Error checking for new files: {e}")
            # When in doubt, re-ingest
            force_reingest = True
    
    if force_reingest:
        print(f"[ingest] {'='*50}")
        print(f"[ingest] STARTING DOCUMENT INGESTION")
        print(f"[ingest] {'='*50}")
        
        docs = load_all_documents()
        if docs:
            chunks = split_into_chunks(docs)
            print("[ingest] Filtering complex metadata for Chroma compatibility...")
            chunks = filter_complex_metadata(chunks)
            vectordb = build_vectorstore(chunks)
            
            if vectordb:
                # Save record of what was ingested
                try:
                    current_files = sorted([f.name for f in Path(DOCS_DIR).iterdir() if f.is_file()])
                    record_file = os.path.join(DB_DIR, "ingestion_record.txt")
                    with open(record_file, 'w', encoding='utf-8') as f:
                        f.write('\n'.join(current_files))
                    print(f"[ingest] ✅ Saved ingestion record for {len(current_files)} files.")
                except Exception as e:
                    print(f"[ingest] ⚠ Could not save ingestion record: {e}")
                
                print(f"[ingest] ✅ Ingestion complete!")
            else:
                print(f"[ingest] ❌ Ingestion failed - could not build vector store.")
        else:
            print(f"[ingest] ❌ No documents to ingest.")
    else:
        print(f"[ingest] ✅ DB already exists with up-to-date content → Skipping ingestion.")

if __name__ == "__main__":
    run_ingestion_if_needed()