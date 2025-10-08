#backend/db/db_utils.py

import sqlite3
from typing import List, Tuple, Optional

DB_PATH = "backend/db/operators.db"

def get_connection():
    """Return a SQLite connection."""
    return sqlite3.connect(DB_PATH)

def fetch_operators() -> List[Tuple]:
    """Fetch all operators."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, email, ward_id FROM operators")
    data = cursor.fetchall()
    conn.close()
    return data

def fetch_operator_by_email(email: str) -> Optional[Tuple]:
    """Fetch operator by email."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, email, ward_id FROM operators WHERE email=?", (email,))
    operator = cursor.fetchone()
    conn.close()
    return operator
