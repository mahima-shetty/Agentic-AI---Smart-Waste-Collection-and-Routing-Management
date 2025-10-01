import streamlit as st
import folium
from streamlit_folium import st_folium
import requests

# FastAPI endpoint
API_URL = "http://127.0.0.1:8000/bins"

st.title("Smart Waste Management Dashboard")

# Fetch data from FastAPI
try:
    response = requests.get(API_URL)
    response.raise_for_status()
    data = response.json()
except Exception as e:
    st.error(f"Error fetching data from API: {e}")
    data = {"garbage_bins": [], "dumping_grounds": []}

# Sidebar KPIs
st.sidebar.header("Dashboard KPIs")
st.sidebar.write(f"Total Garbage Bins: {len(data['garbage_bins'])}")
st.sidebar.write(f"Total Dumping Grounds: {len(data['dumping_grounds'])}")

# Placeholder for AI suggestions
st.sidebar.header("AI Suggestions")
st.sidebar.write("• Placeholder: Bins to collect soon")
st.sidebar.write("• Placeholder: Overflow alerts")

# Create folium map
m = folium.Map(location=[19.07, 72.88], zoom_start=12)

# Add dumping grounds
for d in data.get("dumping_grounds", []):
    folium.Marker(
        location=[d["latitude"], d["longitude"]],
        popup=d["name"],
        icon=folium.Icon(color='green', icon='leaf')
    ).add_to(m)

# Add garbage bins
for b in data.get("garbage_bins", []):
    folium.Marker(
        location=[b["latitude"], b["longitude"]],
        popup=b["name"],
        icon=folium.Icon(color='gray', icon='trash')
    ).add_to(m)

# Display map
st_folium(m, width=700, height=500)
