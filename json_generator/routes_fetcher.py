# json_generator/routes_fetcher.py
import osmnx as ox
import json

# Fetch Mumbai driving network
G = ox.graph_from_place("Mumbai, India", network_type="drive")
# G = ox.simplify_graph(G)  <-- REMOVE this line

# Keep only major + secondary roads
major_highways = ["motorway", "trunk", "primary", "secondary"]
edges = []
for u, v, k, data in G.edges(keys=True, data=True):
    if isinstance(data.get("highway"), list):
        highway_type = data["highway"][0]
    else:
        highway_type = data.get("highway")
    if highway_type in major_highways:
        # Use geometry if exists, else fallback to endpoints
        if "geometry" in data:
            coords = [(lat, lon) for lon, lat in data["geometry"].coords]
        else:
            coords = [(G.nodes[u]["y"], G.nodes[u]["x"]), (G.nodes[v]["y"], G.nodes[v]["x"])]
        edges.append({"id": f"{u}-{v}-{k}", "coords": coords})

# Save to JSON
with open("data/all_routes.json", "w") as f:
    json.dump(edges, f, indent=2)
