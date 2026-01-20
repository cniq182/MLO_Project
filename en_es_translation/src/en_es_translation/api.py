from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
import torch

from .utils import load_model


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
def translate(request: TranslationRequest) -> str:
    model = app.state.model
    
    with torch.no_grad():
        print(model)
        result = model([request.text])[0]
    
    return result