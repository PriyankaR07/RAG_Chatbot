# ingest.py - UPDATED with Excel file_type fix
from pathlib import Path
import os
import re
from collections import defaultdict
import textwrap
import logging
import pandas as pd

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
                    docs.append(Document(page_content=text, metadata={"source": Path(pdf_path).name, "file_type": "pdf"}))
    except Exception as e:
        print(f"[ingest][ocr] Failed to OCR {pdf_path}: {e}")
    print(f"[ingest][ocr] OCR produced {len(docs)} page documents for {pdf_path}")
    return docs

def load_excel_enhanced(file_path):
    """ENHANCED: Load Excel with rich content extraction and proper file_type metadata"""
    documents = []
    
    try:
        file_name = Path(file_path).name
        print(f"[ingest][excel] Loading Excel file → {file_name}")
        
        # Read all sheets
        excel_file = pd.ExcelFile(file_path)
        print(f"[ingest][excel]   Found {len(excel_file.sheet_names)} sheet(s): {excel_file.sheet_names}")
        
        for sheet_name in excel_file.sheet_names:
            try:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                
                if df.empty:
                    print(f"[ingest][excel]   ⚠ Sheet '{sheet_name}' is empty, skipping")
                    continue
                
                print(f"[ingest][excel]   ✓ Sheet '{sheet_name}': {len(df)} rows × {len(df.columns)} columns")
                
                # Build rich, searchable content
                content_parts = []
                
                # Header
                content_parts.append(f"{'='*60}")
                content_parts.append(f"EXCEL FILE: {file_name}")
                content_parts.append(f"SHEET: {sheet_name}")
                content_parts.append(f"{'='*60}\n")
                
                # Basic info
                content_parts.append(f"📊 DIMENSIONS: {len(df)} rows × {len(df.columns)} columns\n")
                
                # Column information
                content_parts.append(f"📋 COLUMNS ({len(df.columns)}):")
                for i, col in enumerate(df.columns, 1):
                    dtype = str(df[col].dtype)
                    non_null = df[col].count()
                    content_parts.append(f"  {i}. {col} (type: {dtype}, non-null: {non_null}/{len(df)})")
                content_parts.append("")
                
                # Numeric summaries
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                if numeric_cols:
                    content_parts.append("📈 NUMERIC DATA SUMMARY:")
                    for col in numeric_cols:
                        stats = df[col].describe()
                        content_parts.append(
                            f"  • {col}: min={stats['min']:.2f}, max={stats['max']:.2f}, "
                            f"mean={stats['mean']:.2f}, median={stats['50%']:.2f}"
                        )
                    content_parts.append("")
                
                # Categorical summaries
                categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
                if categorical_cols:
                    content_parts.append("📝 CATEGORICAL DATA SUMMARY:")
                    for col in categorical_cols[:5]:  # First 5 categorical
                        unique_count = df[col].nunique()
                        content_parts.append(f"  • {col}: {unique_count} unique values")
                        if unique_count <= 10:
                            top_vals = df[col].value_counts().head(5).to_dict()
                            for val, count in top_vals.items():
                                content_parts.append(f"    - '{val}': {count} occurrences")
                    content_parts.append("")
                
                # Sample data (first 20 rows)
                content_parts.append("📄 SAMPLE DATA (First 20 rows):")
                content_parts.append(df.head(20).to_string(index=True))
                content_parts.append("")
                
                # Row-by-row searchable format (ALL rows)
                content_parts.append("🔍 DETAILED ROW DATA (All rows):")
                for idx, row in df.iterrows():
                    row_parts = []
                    for col in df.columns:
                        val = row[col]
                        if pd.notna(val):  # Skip NaN
                            row_parts.append(f"{col}={val}")
                    if row_parts:
                        content_parts.append(f"Row {idx}: {', '.join(row_parts)}")
                
                full_content = "\n".join(content_parts)
                
                # Create document with PROPER metadata
                doc = Document(
                    page_content=full_content,
                    metadata={
                        "source": file_name,
                        "file_type": "excel",  # ← CRITICAL FIX: Set file_type to "excel"
                        "sheet_name": sheet_name,
                        "num_rows": len(df),
                        "num_columns": len(df.columns),
                        "columns": ", ".join(str(c) for c in df.columns),
                        "has_numeric": len(numeric_cols) > 0,
                        "has_categorical": len(categorical_cols) > 0
                    }
                )
                
                documents.append(doc)
                print(f"[ingest][excel]     → Created document: {len(full_content):,} characters")
                
            except Exception as e:
                print(f"[ingest][excel]   ❌ Error processing sheet '{sheet_name}': {e}")
        
        print(f"[ingest][excel] ✅ Loaded {len(documents)} sheet(s) from {file_name}\n")
        
    except Exception as e:
        print(f"[ingest][excel] ❌ Error loading Excel file {file_path}: {e}")
        import traceback
        traceback.print_exc()
    
    return documents

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
                    
                    # Clean up the SQL for readability
                    cleaned_content = re.sub(r'\s+', ' ', content).strip()
                    
                    # Extract key information
                    tables_found = re.findall(r'CREATE TABLE\s+(\w+)', content, re.IGNORECASE)
                    inserts_found = re.findall(r'INSERT INTO\s+(\w+)', content, re.IGNORECASE)
                    
                    # Create summary
                    summary = f"SQL FILE: {file.name}\n"
                    summary += f"File Size: {len(content)} characters\n"
                    
                    if tables_found:
                        summary += f"📊 Tables Defined: {', '.join(tables_found)}\n"
                    
                    # Table details
                    table_details = []
                    for table_match in re.finditer(r'CREATE TABLE\s+(\w+)\s*\(([^)]+)\)', content, re.IGNORECASE | re.DOTALL):
                        table_name = table_match.group(1)
                        columns = table_match.group(2)
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
                    
                    # Preview
                    preview = cleaned_content[:2500]
                    if len(cleaned_content) > 2500:
                        preview += "... [truncated]"
                    
                    full_content = f"{summary}\n\n📄 SQL CONTENT PREVIEW:\n```sql\n{preview}\n```"
                    
                    # Create document
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
                    try:
                        simple_doc = Document(
                            page_content=f"SQL File: {file.name}\nUnable to parse content fully.",
                            metadata={"source": file.name, "file_type": "sql"}
                        )
                        docs.append(simple_doc)
                    except:
                        pass
                continue

            # PDF FILES
            if suffix == ".pdf":
                print(f"[ingest] Loading PDF → {file}")
                try:
                    pdf_docs = PyPDFLoader(str(file)).load()
                    non_empty_pages = sum(1 for d in pdf_docs if (d.page_content or "").strip())
                    # If PDF is mostly empty, try OCR
                    if non_empty_pages < max(1, len(pdf_docs) // 2):
                        print("[ingest] PDF text extraction is sparse — attempting OCR fallback.")
                        ocr_docs = ocr_pdf_to_documents(str(file))
                        if ocr_docs:
                            pdf_docs = ocr_docs
                    for d in pdf_docs:
                        d.metadata["source"] = file.name
                        d.metadata["file_type"] = "pdf"
                    docs.extend(pdf_docs)
                    print(f"[ingest] ✓ PDF loaded: {file.name} ({len(pdf_docs)} pages)")
                except Exception as e:
                    print(f"[ingest] Standard PDF load failed, trying OCR: {e}")
                    ocr_docs = ocr_pdf_to_documents(str(file))
                    docs.extend(ocr_docs)

            # TEXT FILES
            elif suffix == ".txt":
                print(f"[ingest] Loading TXT → {file}")
                txt_docs = TextLoader(str(file), encoding='utf-8', autodetect_encoding=True).load()
                for d in txt_docs:
                    d.metadata["source"] = file.name
                    d.metadata["file_type"] = "txt"
                docs.extend(txt_docs)
                print(f"[ingest] ✓ TXT loaded: {file.name}")

            # CSV FILES
            elif suffix == ".csv":
                print(f"[ingest] Loading CSV → {file}")
                csv_docs = CSVLoader(str(file)).load()
                for d in csv_docs:
                    d.metadata["source"] = file.name
                    d.metadata["file_type"] = "csv"
                docs.extend(csv_docs)
                print(f"[ingest] ✓ CSV loaded: {file.name}")

            # EXCEL FILES - USE ENHANCED LOADER
            elif suffix in [".xls", ".xlsx"]:
                # Use enhanced Excel loader instead of UnstructuredExcelLoader
                excel_docs = load_excel_enhanced(str(file))
                docs.extend(excel_docs)
                # Already prints status inside load_excel_enhanced

            # OTHER DOCUMENT TYPES (Word, PowerPoint, HTML)
            elif suffix in [".doc", ".docx", ".ppt", ".pptx", ".html", ".htm"]:
                print(f"[ingest] Loading {suffix.upper()} (Unstructured) → {file}")
                other_docs = UnstructuredLoader(str(file)).load()
                for d in other_docs:
                    d.metadata["source"] = file.name
                    # Set appropriate file_type
                    if suffix in [".doc", ".docx"]:
                        d.metadata["file_type"] = "word"
                    elif suffix in [".ppt", ".pptx"]:
                        d.metadata["file_type"] = "powerpoint"
                    elif suffix in [".html", ".htm"]:
                        d.metadata["file_type"] = "html"
                docs.extend(other_docs)
                print(f"[ingest] ✓ {suffix} loaded: {file.name}")

            # UNSUPPORTED FILES
            else:
                print(f"[ingest] ⚠ Skipping unsupported file type: {file}")

        except Exception as e:
            print(f"[ingest] ❌ Failed to load {file}: {e}")
            import traceback
            traceback.print_exc()

    # Summary of loaded documents
    if len(docs) == 0:
        print(f"[ingest] ⚠ WARNING: No documents found or extracted in ./{DOCS_DIR} folder!")
    else:
        # Count by file type using the metadata
        file_types = defaultdict(int)
        for doc in docs:
            file_type = doc.metadata.get("file_type", "unknown")
            file_types[file_type] += 1
        
        print(f"\n[ingest] {'='*60}")
        print(f"[ingest] ✅ Loaded {len(docs)} raw document sections.")
        print(f"[ingest] 📊 File type breakdown:")
        for ftype, count in sorted(file_types.items()):
            print(f"[ingest]   - {ftype}: {count} documents")
        print(f"[ingest] {'='*60}\n")
        
    return docs

def split_into_chunks(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)
    print(f"[ingest] Created {len(chunks)} chunks.")
    
    # Show breakdown by file type
    chunk_types = defaultdict(int)
    for chunk in chunks:
        ftype = chunk.metadata.get("file_type", "unknown")
        chunk_types[ftype] += 1
    
    print(f"[ingest] Chunk breakdown by file type:")
    for ftype, count in sorted(chunk_types.items()):
        print(f"[ingest]   - {ftype}: {count} chunks")
    
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
        doc_count = len(collection.get('documents', []))
        print(f"[ingest] Verification: Chroma DB contains {doc_count} document chunks.")
        
        # Check Excel chunks specifically
        metadatas = collection.get('metadatas', [])
        excel_chunks = sum(1 for m in metadatas if m and m.get('file_type') == 'excel')
        if excel_chunks > 0:
            print(f"[ingest] ✓ Excel chunks in database: {excel_chunks}")
        
        return vectordb
    except Exception as e:
        print(f"[ingest] ❌ Failed to build Chroma DB: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_ingestion_if_needed():
    """Run ingestion only if needed (new files or DB doesn't exist)"""
    
    # Check if docs directory exists
    if not os.path.exists(DOCS_DIR):
        print(f"[ingest] ❌ Docs directory '{DOCS_DIR}' does not exist. Creating it...")
        os.makedirs(DOCS_DIR, exist_ok=True)
        print(f"[ingest] ✅ Created '{DOCS_DIR}'. Please add documents and restart.")
        return

    # Check if there are any files
    files = list(Path(DOCS_DIR).iterdir())
    if not files:
        print(f"[ingest] ⚠ No files found in '{DOCS_DIR}'. Please add documents.")
        return
    
    force_reingest = False
    
    # Check if DB exists
    if not os.path.exists(DB_DIR):
        print(f"[ingest] No DB directory '{DB_DIR}' found → Running ingestion...")
        force_reingest = True
    elif not os.listdir(DB_DIR):
        print(f"[ingest] DB directory '{DB_DIR}' is empty → Running ingestion...")
        force_reingest = True
    else:
        try:
            current_files = sorted([f.name for f in Path(DOCS_DIR).iterdir() if f.is_file()])
            record_file = os.path.join(DB_DIR, "ingestion_record.txt")
            
            if os.path.exists(record_file):
                with open(record_file, 'r', encoding='utf-8') as f:
                    previous_files = f.read().splitlines()
                
                if set(current_files) != set(previous_files):
                    print(f"[ingest] 📦 File changes detected → Running ingestion...")
                    force_reingest = True
                else:
                    print(f"[ingest] ✅ DB is up-to-date with current files.")
            else:
                print(f"[ingest] No ingestion record found → Running ingestion...")
                force_reingest = True
                
        except Exception as e:
            print(f"[ingest] ⚠ Error checking files: {e}")
            force_reingest = True
    
    if force_reingest:
        print(f"\n[ingest] {'='*60}")
        print(f"[ingest] STARTING DOCUMENT INGESTION")
        print(f"[ingest] {'='*60}\n")
        
        docs = load_all_documents()
        if docs:
            chunks = split_into_chunks(docs)
            print("[ingest] Filtering complex metadata for Chroma compatibility...")
            chunks = filter_complex_metadata(chunks)
            vectordb = build_vectorstore(chunks)
            
            if vectordb:
                # Save ingestion record
                try:
                    current_files = sorted([f.name for f in Path(DOCS_DIR).iterdir() if f.is_file()])
                    record_file = os.path.join(DB_DIR, "ingestion_record.txt")
                    with open(record_file, 'w', encoding='utf-8') as f:
                        f.write('\n'.join(current_files))
                    print(f"[ingest] ✅ Saved ingestion record for {len(current_files)} files.")
                except Exception as e:
                    print(f"[ingest] ⚠ Could not save ingestion record: {e}")
                
                print(f"\n[ingest] {'='*60}")
                print(f"[ingest] ✅ INGESTION COMPLETE!")
                print(f"[ingest] {'='*60}\n")
            else:
                print(f"[ingest] ❌ Ingestion failed - could not build vector store.")
        else:
            print(f"[ingest] ❌ No documents to ingest.")
    else:
        print(f"[ingest] ✅ DB already exists with up-to-date content → Skipping ingestion.")

if __name__ == "__main__":
    import sys
    
    # Check for --force flag
    if "--force" in sys.argv or "-f" in sys.argv:
        print("[ingest] 🔄 FORCE RE-INGESTION requested via command line flag")
        print("[ingest] Deleting existing database...\n")
        if os.path.exists(DB_DIR):
            import shutil
            shutil.rmtree(DB_DIR)
            print("[ingest] ✓ Deleted old database\n")
    
    run_ingestion_if_needed()