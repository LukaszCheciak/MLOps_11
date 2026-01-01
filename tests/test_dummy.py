import pytest
import numpy as np
from fastapi.testclient import TestClient
from unittest.mock import MagicMock
import sentiment_app.app as app_module

client = TestClient(app_module.app)

@pytest.fixture(autouse=True)
def setup_fake_models():
    fake_tokenizer = MagicMock()
    fake_embedding_session = MagicMock()
    fake_classifier_session = MagicMock()

    fake_encoded = MagicMock()
    fake_encoded.ids = [101, 2055, 102]
    fake_encoded.attention_mask = [1, 1, 1]
    fake_tokenizer.encode.return_value = fake_encoded

    fake_embedding_session.run.return_value = [np.zeros((1, 384), dtype=np.float32)]

    fake_logits = np.array([[0.1, 0.9]], dtype=np.float32)
    fake_classifier_session.run.return_value = [fake_logits]
    
    fake_input_node = MagicMock()
    fake_input_node.name = "float_input"
    fake_classifier_session.get_inputs.return_value = [fake_input_node]

    app_module.tokenizer = fake_tokenizer
    app_module.embedding_session = fake_embedding_session
    app_module.classifier_session = fake_classifier_session

def test_api_predict_positive_flow():
    test_payload = {"text": "MLOps with GitHub Actions is quite powerful."}
    response = client.post("/predict", json=test_payload)
    
    assert response.status_code == 200
    data = response.json()
    assert data["text"] == test_payload["text"]
    assert data["sentiment"] == "positive"
    assert isinstance(data["confidence"], float)

def test_api_validation_error():
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}