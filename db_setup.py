# db_setup.py
from pathlib import Path
from sqlalchemy import create_engine, text
import os
import re

DOCS_DIR = "docs"
DB_FILE = "agent_data.db"
DB_URI = f"sqlite:///{DB_FILE}"

def _sanitize_mysql_for_sqlite(sql_script: str) -> str:
    """Convert MySQL syntax to SQLite-compatible SQL"""
    
    # 1. Remove MySQL-specific comments
    script = re.sub(r'/\*!.*?\*/;', '', sql_script, flags=re.DOTALL)
    script = re.sub(r'^--.*$', '', script, flags=re.MULTILINE)
    script = re.sub(r'#.*$', '', script, flags=re.MULTILINE)
    
    # 2. Remove backticks
    script = script.replace('`', '')
    
    # 3. Fix data types (handle variations: 'int(11)', 'int (11)', 'int( 11 )')
    script = re.sub(r'\b(tiny|small|medium|big)?int\s*\(\s*\d+\s*\)', 'INTEGER', script, flags=re.IGNORECASE)
    script = re.sub(r'\bdouble(\s*\(\s*\d+\s*,\s*\d+\s*\))?', 'REAL', script, flags=re.IGNORECASE)
    script = re.sub(r'\bfloat(\s*\(\s*\d+\s*,\s*\d+\s*\))?', 'REAL', script, flags=re.IGNORECASE)
    script = re.sub(r'\bdecimal\s*\(\s*\d+\s*,\s*\d+\s*\)', 'REAL', script, flags=re.IGNORECASE)
    
    # 4. Convert VARCHAR/CHAR to TEXT
    script = re.sub(r'\bvarchar\s*\(\s*\d+\s*\)', 'TEXT', script, flags=re.IGNORECASE)
    script = re.sub(r'\bchar\s*\(\s*\d+\s*\)', 'TEXT', script, flags=re.IGNORECASE)
    
    # 5. Remove MySQL table options
    script = re.sub(r'\)\s*ENGINE\s*=\s*\w+', ')', script, flags=re.IGNORECASE)
    script = re.sub(r'\s*AUTO_INCREMENT\s*=\s*\d+', '', script, flags=re.IGNORECASE)
    script = re.sub(r'\s*DEFAULT\s+CHARSET\s*=\s*\w+', '', script, flags=re.IGNORECASE)
    script = re.sub(r'\s*COLLATE\s*=\s*\w+', '', script, flags=re.IGNORECASE)
    
    # 6. Remove LOCK/UNLOCK statements
    script = re.sub(r'(LOCK|UNLOCK)\s+TABLES.*?;', '', script, flags=re.IGNORECASE | re.DOTALL)
    
    # 7. Fix AUTO_INCREMENT on column definitions
    script = re.sub(r'\bAUTO_INCREMENT\b', 'AUTOINCREMENT', script, flags=re.IGNORECASE)
    
    # 8. Remove UNSIGNED keyword
    script = re.sub(r'\bUNSIGNED\b', '', script, flags=re.IGNORECASE)
    
    # 9. Clean up extra semicolons and whitespace
    script = re.sub(r';\s*;', ';', script)
    script = re.sub(r'\n\s*\n', '\n', script)
    
    return script

def create_database_from_sql_files(db_uri: str = DB_URI):
    """Create SQLite database from .sql files in docs directory"""
    docs_path = Path(DOCS_DIR)
    
    # Ensure docs directory exists
    if not docs_path.exists():
        print(f"[DB Setup] Creating {DOCS_DIR} directory...")
        docs_path.mkdir(exist_ok=True)
    
    # Reset primary DB for fresh start
    if os.path.exists(DB_FILE):
        try:
            os.remove(DB_FILE)
            print(f"[DB Setup] ✓ Deleted old {DB_FILE} for fresh start")
        except Exception as e:
            print(f"[DB Setup] ⚠ Could not delete old DB: {e}")

    # Create engine with better error handling
    try:
        engine = create_engine(db_uri, echo=False)
        print(f"[DB Setup] Created engine for {db_uri}")
    except Exception as e:
        print(f"[DB Setup] ❌ Failed to create engine: {e}")
        return None
    
    # Find all .sql files
    sql_files = sorted(docs_path.glob("*.sql"))
    
    if not sql_files:
        print(f"[DB Setup] ℹ No .sql files found in {DOCS_DIR}/")
        print(f"[DB Setup] Creating empty database at {DB_FILE}")
        # Create empty database
        try:
            with engine.connect() as connection:
                connection.execute(text("SELECT 1"))
                connection.commit()
            return db_uri
        except Exception as e:
            print(f"[DB Setup] ❌ Failed to create empty DB: {e}")
            return None
    
    print(f"[DB Setup] Found {len(sql_files)} SQL file(s): {[f.name for f in sql_files]}")
    
    # Process each SQL file
    try:
        with engine.connect() as connection:
            # Enable foreign keys
            connection.execute(text("PRAGMA foreign_keys = ON;"))
            
            total_success = 0
            total_statements = 0
            
            for sql_file in sql_files:
                print(f"\n[DB Setup] Processing: {sql_file.name}")
                
                try:
                    # Read file with UTF-8 BOM handling
                    raw_sql = sql_file.read_text(encoding='utf-8-sig')
                    
                    # Sanitize for SQLite
                    clean_sql = _sanitize_mysql_for_sqlite(raw_sql)
                    
                    # Split into statements (handle multiline)
                    statements = re.split(r';\s*(?=\n|$)', clean_sql, flags=re.MULTILINE)
                    statements = [s.strip() for s in statements if s.strip()]
                    
                    print(f"[DB Setup] Found {len(statements)} SQL statements in {sql_file.name}")
                    
                    success_count = 0
                    for i, stmt in enumerate(statements, 1):
                        if not stmt or len(stmt) < 5:
                            continue
                            
                        total_statements += 1
                        
                        try:
                            connection.execute(text(stmt))
                            success_count += 1
                            total_success += 1
                        except Exception as stmt_error:
                            # Only show first few errors to avoid spam
                            if success_count < 3:
                                print(f"   ⚠ Statement {i} failed: {str(stmt_error)[:100]}")
                    
                    print(f"[DB Setup] ✓ Imported {success_count}/{len(statements)} statements from {sql_file.name}")
                    
                except Exception as file_error:
                    print(f"[DB Setup] ❌ Failed to process {sql_file.name}: {file_error}")
                    continue
            
            # Commit all changes
            connection.commit()
            print(f"\n[DB Setup] ✅ COMPLETE: {total_success}/{total_statements} statements imported successfully")
            
            # Verify tables were created
            result = connection.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
            tables = [row[0] for row in result]
            print(f"[DB Setup] Created {len(tables)} tables: {tables}")
            
    except Exception as e:
        print(f"[DB Setup] ❌ Database setup failed: {e}")
        return None
    
    return db_uri

if __name__ == "__main__":
    result = create_database_from_sql_files(DB_URI)
    if result:
        print(f"\n✅ Database ready at: {result}")
    else:
        print("\n❌ Database setup failed")