from fastapi import FastAPI
from fastapi.responses import JSONResponse
import json
from pathlib import Path

app = FastAPI()

DATA_FILE = Path(__file__).parent.parent / "data" / "markers.json"

@app.get("/")
def root():
    return {"message": "Smart Waste Management API is running"}

@app.get("/bins")
def get_bins():
    try:
        with open(DATA_FILE, "r") as f:
            data = json.load(f)
        return JSONResponse(content=data)
    except Exception as e:
        return {"error": str(e)}
