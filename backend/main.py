from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import json
from pathlib import Path
from urllib.parse import unquote
import random
import threading
import time

app = FastAPI()

# Enable CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

DATA_FILE = Path(__file__).parent.parent / "data" / "markers.json"

# ---------------- Utility functions ----------------

def load_data():
    with open(DATA_FILE, "r") as f:
        return json.load(f)

def save_data(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=4)

def update_bin_fill():
    """Simulate bin fill levels incrementing every 5 minutes"""
    data = load_data()
    changed = False
    for b in data["garbage_bins"]:
        if "fill_level" not in b:
            b["fill_level"] = random.randint(0, 20)  # start with some fill
            changed = True
        elif b["fill_level"] < 100:
            b["fill_level"] += random.randint(1, 5)
            if b["fill_level"] > 100:
                b["fill_level"] = 100
            changed = True
    if changed:
        save_data(data)

def start_fill_simulator():
    """Background thread to update bins every 5 minutes"""
    def run():
        while True:
            update_bin_fill()
            time.sleep(10)  # 10 secs
    t = threading.Thread(target=run, daemon=True)
    t.start()

# Start simulator on server startup
start_fill_simulator()

# ------------------- Endpoints -------------------

@app.get("/")
def root():
    return {"message": "Smart Waste Management API is running"}

@app.get("/bins")
def get_bins():
    try:
        data = load_data()
        return JSONResponse(content=data)
    except Exception as e:
        return {"error": str(e)}

@app.post("/bins/add")
def add_bin(bin: dict):
    data = load_data()
    if "name" not in bin or "latitude" not in bin or "longitude" not in bin:
        raise HTTPException(status_code=400, detail="Bin must have name, latitude, longitude")
    if any(b["name"].lower() == bin["name"].lower() for b in data["garbage_bins"]):
        raise HTTPException(status_code=400, detail="Bin with this name already exists")
    data["garbage_bins"].append(bin)
    save_data(data)
    return {"message": f"{bin['name']} added successfully"}

@app.delete("/bins/delete/{bin_name}")
def delete_bin(bin_name: str):
    bin_name = unquote(bin_name).strip()
    data = load_data()
    new_bins = [b for b in data["garbage_bins"] if b["name"].lower() != bin_name.lower()]
    if len(new_bins) == len(data["garbage_bins"]):
        raise HTTPException(status_code=404, detail=f"Bin '{bin_name}' not found")
    data["garbage_bins"] = new_bins
    save_data(data)
    return {"message": f"{bin_name} deleted successfully"}
