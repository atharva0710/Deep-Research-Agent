import sqlite3
import json
import uuid
from datetime import datetime, timezone
from typing import List, Dict, Any

DB_PATH = "research_sessions.db"

def init_db():
    """Initializes the SQLite schema for session management."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Sessions table stores high-level info and rolling summary
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            created_at TEXT,
            summary TEXT
        )
    ''')
    
    # Turns table stores detailed interaction history
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS turns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            query TEXT,
            search_queries TEXT,
            opened_urls TEXT,
            selected_snippets TEXT,
            final_answer TEXT,
            timestamp TEXT,
            FOREIGN KEY(session_id) REFERENCES sessions(session_id)
        )
    ''')
    
    # Messages table stores independent conversation history messages
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            role TEXT,
            content TEXT,
            timestamp TEXT,
            FOREIGN KEY(session_id) REFERENCES sessions(session_id)
        )
    ''')
    
    conn.commit()
    conn.close()

def create_session() -> str:
    """Creates a new session and returns the session_id."""
    init_db()
    session_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('INSERT INTO sessions (session_id, created_at, summary) VALUES (?, ?, ?)',
                   (session_id, timestamp, ""))
    conn.commit()
    conn.close()
    
    return session_id

def save_turn(session_id: str, query: str, search_queries: List[str], opened_urls: List[str], 
              selected_snippets: List[str], final_answer: str):
    """Saves a single research turn and individual messages to the database."""
    timestamp = datetime.now(timezone.utc).isoformat()
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Save the turn details
    cursor.execute('''
        INSERT INTO turns (session_id, query, search_queries, opened_urls, selected_snippets, final_answer, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        session_id,
        query,
        json.dumps(search_queries),
        json.dumps(opened_urls),
        json.dumps(selected_snippets),
        final_answer,
        timestamp
    ))
    
    # Save separate user query message
    cursor.execute('''
        INSERT INTO messages (session_id, role, content, timestamp)
        VALUES (?, ?, ?, ?)
    ''', (session_id, 'user', query, timestamp))
    
    # Save separate assistant answer message
    cursor.execute('''
        INSERT INTO messages (session_id, role, content, timestamp)
        VALUES (?, ?, ?, ?)
    ''', (session_id, 'assistant', final_answer, timestamp))
    
    conn.commit()
    conn.close()

def get_session_history(session_id: str, limit: int = 3) -> List[Dict[str, Any]]:
    """Retrieves the recent history of a session, up to `limit` turns."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT query, final_answer FROM turns 
        WHERE session_id = ? 
        ORDER BY id DESC LIMIT ?
    ''', (session_id, limit))
    
    rows = cursor.fetchall()
    conn.close()
    
    # Return in chronological order
    history = []
    for row in reversed(rows):
        history.append({
            "query": row["query"],
            "final_answer": row["final_answer"]
        })
        
    return history

def update_session_summary(session_id: str, new_summary: str):
    """Updates the rolling summary for a session."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('UPDATE sessions SET summary = ? WHERE session_id = ?', (new_summary, session_id))
    conn.commit()
    conn.close()
    
def get_session_summary(session_id: str) -> str:
    """Retrieves the rolling summary."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT summary FROM sessions WHERE session_id = ?', (session_id,))
    row = cursor.fetchone()
    conn.close()
    return row[0] if row and row[0] else ""
