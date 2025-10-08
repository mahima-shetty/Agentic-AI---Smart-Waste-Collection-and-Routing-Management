# dashboard/auth.py
from backend.db.db_utils import fetch_operator_by_email

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
