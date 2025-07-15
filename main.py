from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import os

app = FastAPI()

# ✅ Remplace ceci par ton propre token Hugging Face
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_ID = "EduBot-v1/falcon-ibam"  # remplace avec ton modèle

# Base URL de l'API Hugging Face Inference
API_URL = f"https://api-inference.huggingface.co/models/{MODEL_ID}"
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

class Prompt(BaseModel):
    prompt: str

@app.post("/generate")
async def generate(prompt: Prompt):
    response = requests.post(API_URL, headers=HEADERS, json={"inputs": prompt.prompt})
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.text)
    result = response.json()
    # Extraire le texte généré
    generated_text = result[0]["generated_text"] if isinstance(result, list) else result
    return {"response": generated_text}
