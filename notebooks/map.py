import folium

# Create a map centered at a specific location with a chosen zoom level
m = folium.Map(location=[19.0824822, 72.7141282], zoom_start=12) #Mumbai

# Add a marker to the map

folium.Marker(
    location=[19.172972872270734, 72.97674663940518],  # Dumping Ground 1
    popup="Mulund Dumping Ground 1",
    icon=folium.Icon(color="green", icon="leaf", prefix='fa')
).add_to(m)

folium.Marker(
    location=[19.123231972026957, 72.95404179025559],  # Dumping Ground 2
    popup="BMC Dumping Ground 2",
    icon=folium.Icon(color="green", icon="leaf", prefix='fa')
).add_to(m)


folium.Marker(
    location=[19.071892501022422, 72.92989606399117],  # Dumping Ground 3
    popup="Deonar Dumping Ground 3",
    icon=folium.Icon(color="green", icon="leaf", prefix='fa')
).add_to(m)

#########################################################################

# Garbage bins


folium.Marker(
    location=[19.168254558644758, 72.94358994398651],  # Bins 1
    popup="Dustbin 1",
    icon=folium.Icon(color="gray", icon="dumpster", prefix='fa')
).add_to(m)

folium.Marker(
    location=[19.151482656453155, 72.93089775714564],  # Bins 2
    popup="Dustbin 2",
    icon=folium.Icon(color="gray", icon="dumpster", prefix='fa')
).add_to(m)

folium.Marker(
    location=[19.133186041131196, 72.91972681628498],  # Bins 3
    popup="Dustbin 3",
    icon=folium.Icon(color="gray", icon="dumpster", prefix='fa')
).add_to(m)


folium.Marker(
    location=[19.110947837238985, 72.90388294346644],  # Bins 4
    popup="Dustbin 4",
    icon=folium.Icon(color="gray", icon="dumpster", prefix='fa')
).add_to(m)


folium.Marker(
    location=[19.090620776292223, 72.89539585798067],  # Bins 5
    popup="Dustbin 5",
    icon=folium.Icon(color="gray", icon="dumpster", prefix='fa')
).add_to(m)


folium.Marker(
    location=[19.074279678172953, 72.8824762991172],  # Bins 6
    popup="Dustbin 6",
    icon=folium.Icon(color="gray", icon="dumpster", prefix='fa')
).add_to(m)


folium.Marker(
    location=[19.053765756330648, 72.87299399035572],  # Bins 7
    popup="Dustbin 7",
    icon=folium.Icon(color="gray", icon="dumpster", prefix='fa')
).add_to(m)


folium.Marker(
    location=[19.044343841645652, 72.84959187636798],  # Bins 8
    popup="Dustbin 8",
    icon=folium.Icon(color="gray", icon="dumpster", prefix='fa')
).add_to(m)


folium.Marker(
    location=[19.01994163371838, 72.83524834670496],  # Bins 9
    popup="Dustbin 9",
    icon=folium.Icon(color="gray", icon="dumpster", prefix='fa')
).add_to(m)

folium.Marker(
    location=[18.99834970327438, 72.82109474837868],  # Bins 10
    popup="Dustbin 10",
    icon=folium.Icon(color="gray", icon="dumpster", prefix='fa')
    
).add_to(m)


folium.Marker(
    location=[18.972468914623537, 72.81590357906927],  # Bins 11
    popup="Dustbin 11",
    icon=folium.Icon(color="gray", icon="dumpster", prefix='fa')
).add_to(m)

folium.Marker(
    location=[18.95575243894783, 72.83563729400159],  # Bins 12
    popup="Dustbin 12",
    icon=folium.Icon(color="gray", icon="dumpster", prefix='fa')
).add_to(m)



# Save the map as an HTML file
m.save("my_interactive_map.html")