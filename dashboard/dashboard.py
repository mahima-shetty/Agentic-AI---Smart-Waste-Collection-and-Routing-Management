# dashboard/dashboard.py
import sys
import os

# Get absolute path to the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add dashboard folder (for local imports)
sys.path.append(current_dir)

# Add project root (one level up) so backend modules are found
sys.path.append(os.path.dirname(current_dir))


from ward_panels import get_ward_stats
from map_view import generate_ward_map

def render_dashboard():
    """Render the main dashboard (CLI demo version)."""
    print("=== WARD DASHBOARD ===\n")
    
    stats = get_ward_stats()
    for s in stats:
        print(f"Ward {s['ward_id']} - Operator: {s['operator_name']}")
        print(f"Issues Reported: {s['issues_reported']}, Resolved: {s['issues_resolved']}\n")
    
    print("Generating ward map...")
    m = generate_ward_map()
    m.save("ward_map.html")
    print("Map saved as ward_map.html")

if __name__ == "__main__":
    render_dashboard()
