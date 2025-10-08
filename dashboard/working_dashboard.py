# dashboard/working_dashboard.py
import streamlit as st
import folium
from streamlit_folium import st_folium
import requests
import json
import random
from pathlib import Path
from streamlit_autorefresh import st_autorefresh

BINS_API_URL = "http://127.0.0.1:8000/bins"
ROUTES_FILE = Path("data/all_routes.json")

st.title("Smart Waste Management Dashboard")

# ---------------- Auto-refresh every 30 secs ----------------
count = st_autorefresh(interval=300_000, limit=None, key="refresh")

# ---------------- Fetch Data Functions ----------------
def fetch_bins():
    try:
        res = requests.get(BINS_API_URL)
        res.raise_for_status()
        return res.json()
    except Exception as e:
        st.error(f"Error fetching bins: {e}")
        return {"garbage_bins": [], "dumping_grounds": []}

def load_routes():
    try:
        with open(ROUTES_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading routes: {e}")
        return []

# ---------------- Initialize session state ----------------
if "bins_data" not in st.session_state:
    st.session_state.bins_data = fetch_bins()

if "routes_data" not in st.session_state:
    st.session_state.routes_data = load_routes()

if "traffic_colors" not in st.session_state:
    # default all gray
    st.session_state.traffic_colors = {r["id"]: "gray" for r in st.session_state.routes_data}

# ---------------- Add/Delete Bins ----------------
st.sidebar.header("Manage Bins")

# Add a bin
st.sidebar.subheader("Add a Bin")
new_name = st.sidebar.text_input("Bin Name")
new_lat = st.sidebar.number_input("Latitude", format="%.6f")
new_lon = st.sidebar.number_input("Longitude", format="%.6f")
if st.sidebar.button("Add Bin"):
    if new_name and new_lat and new_lon:
        payload = {"name": new_name, "latitude": new_lat, "longitude": new_lon}
        res = requests.post(f"{BINS_API_URL}/add", json=payload)
        if res.status_code == 200:
            st.success(f"{new_name} added successfully")
            st.session_state.bins_data = fetch_bins()
        else:
            st.error(f"Error: {res.json().get('detail')}")
    else:
        st.warning("Fill all fields to add a bin")

# Delete a bin
st.sidebar.subheader("Delete a Bin")
del_name = st.sidebar.text_input("Bin Name to Delete")
if st.sidebar.button("Delete Bin"):
    if del_name:
        res = requests.delete(f"{BINS_API_URL}/delete/{del_name.strip()}")
        if res.status_code == 200:
            st.success(f"{del_name} deleted successfully")
            st.session_state.bins_data = fetch_bins()
        else:
            st.error(f"Error: {res.json().get('detail')}")
    else:
        st.warning("Enter a bin name to delete")

# ---------------- Sidebar KPIs ----------------
st.sidebar.header("Dashboard KPIs")
st.sidebar.write(f"Total Garbage Bins: {len(st.session_state.bins_data['garbage_bins'])}")
st.sidebar.write(f"Total Dumping Grounds: {len(st.session_state.bins_data['dumping_grounds'])}")
st.sidebar.write(f"Total Routes Monitored: {len(st.session_state.routes_data)}")

# ---------------- Traffic Simulator Button ----------------
if st.sidebar.button("Show Random Traffic"):
    colors = ["green", "yellow", "orange", "red"]
    for route in st.session_state.routes_data:
        st.session_state.traffic_colors[route["id"]] = random.choice(colors)

# ---------------- Map ----------------
m = folium.Map(location=[19.07, 72.88], zoom_start=12)

# Dumping grounds
for d in st.session_state.bins_data.get("dumping_grounds", []):
    folium.Marker(
        location=[d["latitude"], d["longitude"]],
        popup=d["name"],
        icon=folium.Icon(color='green', icon='leaf')
    ).add_to(m)

# Garbage bins with dynamic fill color
for b in st.session_state.bins_data.get("garbage_bins", []):
    fill = b.get("fill_level", 0)
    if fill < 50:
        color = "green"
    elif fill < 80:
        color = "orange"
    else:
        color = "red"
    folium.Marker(
        location=[b["latitude"], b["longitude"]],
        popup=f"{b['name']} - {fill}%",
        icon=folium.Icon(color=color, icon='trash')
    ).add_to(m)

# Draw routes
for route in st.session_state.routes_data:
    folium.PolyLine(
        locations=route["coords"],
        color=st.session_state.traffic_colors[route["id"]],
        weight=3,
        opacity=0.7
    ).add_to(m)

st_folium(m, width=800, height=600)
