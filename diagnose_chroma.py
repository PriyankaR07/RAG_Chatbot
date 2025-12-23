# diagnose_chroma.py
# Run this to see what's actually in your Chroma database

import os
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from collections import defaultdict

CHROMA_DB_DIR = "chroma_db"
EMBEDDING_MODEL = "embeddinggemma:latest"

def diagnose_chroma():
    """Check what's actually stored in Chroma DB"""
    
    print(f"\n{'='*70}")
    print("CHROMA DATABASE DIAGNOSTIC")
    print(f"{'='*70}\n")
    
    if not os.path.exists(CHROMA_DB_DIR):
        print(f"❌ Chroma DB directory '{CHROMA_DB_DIR}' does not exist!")
        print("   Run ingestion first to create the database.")
        return
    
    try:
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        db = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
        
        # Get all documents
        collection = db.get()
        documents = collection.get('documents', [])
        metadatas = collection.get('metadatas', [])
        
        print(f"✓ Total chunks in Chroma DB: {len(documents)}\n")
        
        if len(documents) == 0:
            print("❌ NO DOCUMENTS FOUND IN CHROMA!")
            print("   The database is empty. You need to run ingestion.")
            return
        
        # Analyze by file type
        print(f"{'='*70}")
        print("DOCUMENTS BY FILE TYPE")
        print(f"{'='*70}\n")
        
        by_type = defaultdict(int)
        by_source = defaultdict(int)
        
        for meta in metadatas:
            if meta:
                file_type = meta.get('file_type', 'unknown')
                source = meta.get('source', 'unknown')
                by_type[file_type] += 1
                by_source[os.path.basename(source)] += 1
        
        for file_type, count in sorted(by_type.items()):
            print(f"  {file_type.upper():10} : {count} chunks")
        
        print(f"\n{'='*70}")
        print("DOCUMENTS BY SOURCE FILE")
        print(f"{'='*70}\n")
        
        for source, count in sorted(by_source.items()):
            print(f"  {source:40} : {count} chunks")
        
        # Check for Excel files specifically
        print(f"\n{'='*70}")
        print("EXCEL FILES CHECK")
        print(f"{'='*70}\n")
        
        excel_chunks = [m for m in metadatas if m and m.get('file_type') in ['excel', 'xlsx', 'xls']]
        
        if excel_chunks:
            print(f"✓ Found {len(excel_chunks)} Excel chunks")
            
            # Show Excel sheets
            excel_sheets = defaultdict(list)
            for meta in excel_chunks:
                source = os.path.basename(meta.get('source', 'unknown'))
                sheet = meta.get('sheet_name', 'unknown')
                excel_sheets[source].append(sheet)
            
            print("\nExcel sheets loaded:")
            for source, sheets in excel_sheets.items():
                print(f"  {source}:")
                for sheet in set(sheets):
                    print(f"    - {sheet}")
        else:
            print("❌ NO EXCEL FILES FOUND!")
            print("   This is likely your problem.")
            print("\n   Excel files may not be getting ingested properly.")
            print("   Check:")
            print("   1. Do you have .xlsx or .xls files in your 'docs' folder?")
            print("   2. Is your ingest.py properly handling Excel files?")
            print("   3. Try running the enhanced_ingest.py script")
        
        # Show sample content from each file type
        print(f"\n{'='*70}")
        print("SAMPLE CONTENT FROM EACH FILE TYPE")
        print(f"{'='*70}\n")
        
        samples_shown = defaultdict(int)
        for doc, meta in zip(documents, metadatas):
            if meta:
                file_type = meta.get('file_type', 'unknown')
                if samples_shown[file_type] == 0:  # Show one sample per type
                    source = os.path.basename(meta.get('source', 'unknown'))
                    preview = doc[:200].replace('\n', ' ')
                    print(f"[{file_type.upper()}] From {source}:")
                    print(f"  {preview}...")
                    print()
                    samples_shown[file_type] += 1
        
        # Test search for Excel content
        print(f"\n{'='*70}")
        print("TEST SEARCH FOR EXCEL CONTENT")
        print(f"{'='*70}\n")
        
        test_queries = [
            "excel data",
            "sheet",
            "rows columns",
            "spreadsheet"
        ]
        
        for query in test_queries:
            results = db.similarity_search(query, k=3)
            excel_results = [r for r in results if r.metadata.get('file_type') in ['excel', 'xlsx', 'xls']]
            print(f"Query: '{query}' → Found {len(excel_results)} Excel chunks out of {len(results)} total")
        
        print(f"\n{'='*70}\n")
        
    except Exception as e:
        print(f"❌ Error accessing Chroma DB: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    diagnose_chroma()