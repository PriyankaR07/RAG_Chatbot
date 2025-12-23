# sql_agent.py
import sqlite3
import re
from pathlib import Path
from langchain_ollama import ChatOllama
import threading

class SQLAgent:
    def __init__(self, db_uri: str, llm_model: str, llm_temperature: float):
        self.db_uri = db_uri
        self.llm_model = llm_model
        self.llm_temperature = llm_temperature
        
        print(f"[SQLAgent] Initializing with model: {llm_model}")
        
        # Extract SQLite file path from URI
        if db_uri.startswith("sqlite:///"):
            self.db_path = db_uri.replace("sqlite:///", "")
        else:
            self.db_path = db_uri
            
        # Initialize LLM for SQL generation
        self.llm = ChatOllama(model=llm_model, temperature=llm_temperature)
        
        # Thread-local storage for connections
        self._thread_local = threading.local()
        
        # Get initial schema
        self._update_schema()
    
    def _get_connection(self):
        """Get thread-safe SQLite connection"""
        if not hasattr(self._thread_local, "connection"):
            try:
                self._thread_local.connection = sqlite3.connect(self.db_path, check_same_thread=False)
                self._thread_local.connection.row_factory = sqlite3.Row
            except Exception as e:
                print(f"[SQLAgent] Failed to create connection: {e}")
                return None
        return self._thread_local.connection
    
    def _update_schema(self):
        """Update schema information from database"""
        conn = self._get_connection()
        if not conn:
            self.schema = {}
            return
            
        cursor = conn.cursor()
        self.schema = {}
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
        tables = [row[0] for row in cursor.fetchall()]
        
        for table in tables:
            # Get column information
            cursor.execute(f"PRAGMA table_info({table})")
            columns = cursor.fetchall()
            
            col_info = []
            for col in columns:
                col_info.append({
                    'name': col[1],
                    'type': col[2],
                    'notnull': col[3],
                    'pk': col[5]
                })
            
            self.schema[table] = col_info
    
    def ask(self, query: str):
        """Main method to handle SQL queries - returns tuple (summary, results)"""
        print(f"[SQLAgent] Processing: {query}")
        
        # Update schema
        self._update_schema()
        
        if not self.schema:
            return "Database has no tables or cannot connect to database", ""
        
        # Simple responses for common queries
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["how many tables", "list tables", "show tables", "tables in"]):
            tables = list(self.schema.keys())
            summary = f"Found {len(tables)} table(s) in the database"
            results = "\n".join([f"- {table}" for table in tables])
            return summary, results
        
        elif any(word in query_lower for word in ["database schema", "database structure", "show schema"]):
            summary = f"Database Schema ({len(self.schema)} tables)"
            results = []
            for table, columns in self.schema.items():
                col_names = [col['name'] for col in columns]
                results.append(f"Table: {table}")
                results.append(f"  Columns: {', '.join(col_names[:5])}")
                if len(col_names) > 5:
                    results.append(f"  (+ {len(col_names)-5} more columns)")
                results.append("")
            return summary, "\n".join(results)
        
        elif "describe" in query_lower or "columns in" in query_lower:
            # Try to find which table they're asking about
            for table in self.schema.keys():
                if table.lower() in query_lower:
                    columns = self.schema[table]
                    col_details = [f"{col['name']} ({col['type']})" for col in columns]
                    summary = f"Table: {table} ({len(columns)} columns)"
                    results = "\n".join([f"- {detail}" for detail in col_details])
                    return summary, results
            
            # If no specific table mentioned, show first table
            first_table = list(self.schema.keys())[0]
            columns = self.schema[first_table]
            col_details = [f"{col['name']} ({col['type']})" for col in columns]
            summary = f"Table: {first_table} ({len(columns)} columns)"
            results = "\n".join([f"- {detail}" for detail in col_details])
            return summary, results
        
        else:
            # Generic response
            tables = list(self.schema.keys())
            summary = f"Database has {len(tables)} table(s)"
            results = f"Available tables:\n" + "\n".join([f"- {table}" for table in tables])
            results += "\n\nAsk about specific tables, columns, or use SQL queries."
            return summary, results
    
    def close(self):
        """Close database connections"""
        if hasattr(self._thread_local, "connection"):
            try:
                self._thread_local.connection.close()
            except:
                pass