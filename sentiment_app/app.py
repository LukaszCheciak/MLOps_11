import os
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI
from pydantic import BaseModel
from tokenizers import Tokenizer

app = FastAPI(title="Sentiment Analysis ONNX API")

SENTIMENT_MAP = {0: "negative", 1: "positive"}

MODEL_DIR = os.getenv("MODEL_DIR", "model")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.json")
EMBEDDING_MODEL_PATH = os.path.join(MODEL_DIR, "onnx_embedding_model.onnx")
CLASSIFIER_MODEL_PATH = os.path.join(MODEL_DIR, "onnx_classifier.onnx")

class PredictRequest(BaseModel):
    text: str

try:
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    embedding_session = ort.InferenceSession(EMBEDDING_MODEL_PATH)
    classifier_session = ort.InferenceSession(CLASSIFIER_MODEL_PATH)
    print("Models load successful!")
except Exception as e:
    print(f"Error loading models: {e}")

@app.post("/predict")
async def predict(request: PredictRequest):

    encoded = tokenizer.encode(request.text)
    
    input_ids = np.array([encoded.ids], dtype=np.int64)
    attention_mask = np.array([encoded.attention_mask], dtype=np.int64)
    
    embedding_inputs = {
        "input_ids": input_ids, 
        "attention_mask": attention_mask
    }
    embeddings = embedding_session.run(None, embedding_inputs)[0]
    
    classifier_input_name = classifier_session.get_inputs()[0].name
    classifier_inputs = {classifier_input_name: embeddings.astype(np.float32)}
    
    prediction_logits = classifier_session.run(None, classifier_inputs)[0]
    
    prediction = int(np.argmax(prediction_logits, axis=1)[0])
    label = SENTIMENT_MAP.get(prediction, "unknown")
    
    return {
        "text": request.text,
        "sentiment": label,
        "confidence": float(np.max(prediction_logits))
    }

@app.get("/health")
def health():
    return {"status": "healthy"}