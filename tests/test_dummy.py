from fastapi.testclient import TestClient
from sentiment_app.app import app

client = TestClient(app)

def test_read_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_predict_sentiment():
    payload = {"text": "I really love this lab!"}
    response = client.post("/predict", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert "sentiment" in data
    assert "confidence" in data
    assert isinstance(data["sentiment"], str)