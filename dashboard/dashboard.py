import streamlit as st
import folium
from streamlit_folium import st_folium
import requests
from streamlit_autorefresh import st_autorefresh

API_URL = "http://127.0.0.1:8000/bins"

st.title("Smart Waste Management Dashboard")

# ---------------- Auto-refresh every 5 mins ----------------
# interval=300000 ms = 5 minutes
count = st_autorefresh(interval=10_000, limit=None, key="refresh")

# ---------------- Fetch Data Function ----------------
def fetch_data():
    try:
        response = requests.get(API_URL)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return {"garbage_bins": [], "dumping_grounds": []}

# Initialize session state for data
st.session_state.data = fetch_data()

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
        res = requests.post(f"{API_URL}/add", json=payload)
        if res.status_code == 200:
            st.success(f"{new_name} added successfully")
            st.session_state.data = fetch_data()  # Refresh data immediately
        else:
            st.error(f"Error: {res.json().get('detail')}")
    else:
        st.warning("Fill all fields to add a bin")

# Delete a bin
st.sidebar.subheader("Delete a Bin")
del_name = st.sidebar.text_input("Bin Name to Delete")
if st.sidebar.button("Delete Bin"):
    if del_name:
        res = requests.delete(f"{API_URL}/delete/{del_name.strip()}")
        if res.status_code == 200:
            st.success(f"{del_name} deleted successfully")
            st.session_state.data = fetch_data()  # Refresh data immediately
        else:
            st.error(f"Error: {res.json().get('detail')}")
    else:
        st.warning("Enter a bin name to delete")

# ---------------- Sidebar KPIs ----------------
st.sidebar.header("Dashboard KPIs")
st.sidebar.write(f"Total Garbage Bins: {len(st.session_state.data['garbage_bins'])}")
st.sidebar.write(f"Total Dumping Grounds: {len(st.session_state.data['dumping_grounds'])}")

# ---------------- Map ----------------
m = folium.Map(location=[19.07, 72.88], zoom_start=12)

# Dumping grounds
for d in st.session_state.data.get("dumping_grounds", []):
    folium.Marker(
        location=[d["latitude"], d["longitude"]],
        popup=d["name"],
        icon=folium.Icon(color='green', icon='leaf')
    ).add_to(m)

# Garbage bins with dynamic fill color
for b in st.session_state.data.get("garbage_bins", []):
    fill = b.get("fill_level", 0)
    # Determine marker color based on fill level
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

st_folium(m, width=700, height=500)
