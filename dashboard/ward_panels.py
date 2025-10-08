# dashboard/ward_panels.py
from backend.db.db_utils import fetch_operators

def get_ward_stats():
    """Return dummy stats for each ward."""
    operators = fetch_operators()
    stats = []
    for op in operators:
        stats.append({
            "ward_id": op[3],
            "operator_name": op[1],
            "issues_reported": 10 * op[3],  # dummy numbers
            "issues_resolved": 8 * op[3]
        })
    return stats
