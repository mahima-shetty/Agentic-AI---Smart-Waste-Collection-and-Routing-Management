# dashboard/auth.py
import streamlit as st
from typing import Optional, Dict, Any
from backend.db.db_utils import fetch_operator_by_email

def authenticate_user(email: str, password: str) -> Optional[Dict[str, Any]]:
    """Authenticate user and return operator info if successful."""
    try:
        operator = fetch_operator_by_email(email)
        if operator and password == get_operator_password(email):
            return {
                "id": operator[0],
                "name": operator[1],
                "email": operator[2],
                "ward_id": operator[3]
            }
        return None
    except Exception as e:
        print(f"Authentication error: {e}")
        return None

def logout_user():
    """Clear user session."""
    keys_to_clear = [
        "authenticated", "operator_id", "operator_name", 
        "operator_email", "ward_id", "current_page"
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

def login(email: str, password: str) -> bool:
    """Simple authentication check."""
    operator = fetch_operator_by_email(email)
    if operator and password == get_operator_password(email):
        return True
    return False

def get_operator_password(email: str) -> str:
    """Fetch password for operator (for demo purposes)."""
    from backend.db.db_utils import get_connection
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT password FROM operators WHERE email=?", (email,))
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else ""
