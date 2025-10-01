from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import json
from pathlib import Path
from urllib.parse import unquote

app = FastAPI()

# Enable CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

DATA_FILE = Path(__file__).parent.parent / "data" / "markers.json"

# Utility functions
def load_data():
    with open(DATA_FILE, "r") as f:
        return json.load(f)

def save_data(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=4)

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
    # Validate input
    if "name" not in bin or "latitude" not in bin or "longitude" not in bin:
        raise HTTPException(status_code=400, detail="Bin must have name, latitude, longitude")
    
    # Avoid duplicate bin names (case-insensitive)
    if any(b["name"].lower() == bin["name"].lower() for b in data["garbage_bins"]):
        raise HTTPException(status_code=400, detail="Bin with this name already exists")

    data["garbage_bins"].append(bin)
    save_data(data)
    return {"message": f"{bin['name']} added successfully"}

@app.delete("/bins/delete/{bin_name}")
def delete_bin(bin_name: str):
    bin_name = unquote(bin_name).strip()  # decode %20 and remove spaces
    data = load_data()
    # Case-insensitive matching
    new_bins = [b for b in data["garbage_bins"] if b["name"].lower() != bin_name.lower()]
    if len(new_bins) == len(data["garbage_bins"]):
        raise HTTPException(status_code=404, detail=f"Bin '{bin_name}' not found")
    data["garbage_bins"] = new_bins
    save_data(data)
    return {"message": f"{bin_name} deleted successfully"}
