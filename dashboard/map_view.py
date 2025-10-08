# dashboard/map_view.py
import folium
from backend.db.db_utils import fetch_operators

def generate_ward_map() -> folium.Map:
    """Create a basic map with dummy coordinates for each ward."""
    m = folium.Map(location=[19.0760, 72.8777], zoom_start=12)  # Mumbai center

    operators = fetch_operators()
    for op in operators:
        ward_id = op[3]
        # Dummy lat/lon, replace with real coordinates later
        lat, lon = 19.0 + 0.01*ward_id, 72.85 + 0.01*ward_id
        folium.Marker([lat, lon], popup=f"Ward {ward_id}: {op[1]}").add_to(m)
    
    return m
