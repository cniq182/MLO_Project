from contextlib import asynccontextmanager
from datetime import datetime
import os
from pathlib import Path
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import torch

from .utils import load_model

# M27: Define the path for the prediction database
# Since api.py is in /src/en_es_translation/, we go up 3 levels to hit MLO_Project root
PROJECT_ROOT = Path(__file__).resolve().parents[3] 
DATABASE_PATH = PROJECT_ROOT / "prediction_database.csv"

def log_to_database(timestamp: str, input_text: str, output_text: str):
    """Writes input/output pairs to a CSV file for drift monitoring."""
    file_exists = DATABASE_PATH.exists()
    
    # Using 'a' (append) mode to add new entries without deleting old ones
    with open(DATABASE_PATH, "a") as f:
        if not file_exists:
            f.write("timestamp,input,prediction\n")
        # Quotes handle text that might contain commas or newlines
        f.write(f'"{timestamp}","{input_text}","{output_text}"\n')

@asynccontextmanager
async def lifespan(app: FastAPI):
    model, device = load_model()
    app.state.model = model
    app.state.device = device
    yield

app = FastAPI(lifespan=lifespan)

class TranslationRequest(BaseModel):
    text: str

@app.post("/translate")
def translate(request: TranslationRequest, background_tasks: BackgroundTasks) -> str:
    model = app.state.model

    with torch.no_grad():
        # Core model logic
        result = model([request.text])[0]

    # M27: Log the interaction in the background
    # This captures the 'When', 'What went in', and 'What came out'
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    background_tasks.add_task(log_to_database, now, request.text, result)

    return result

def main():
    import uvicorn
    # Points to the app inside your package structure
    uvicorn.run("en_es_translation.api:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    main()