import pytest
import numpy as np
from fastapi.testclient import TestClient
from unittest.mock import MagicMock
from sentiment_app.app import app

client = TestClient(app)

@pytest.fixture(autouse=True)
def setup_fake_models():
    fake_tokenizer = MagicMock()
    fake_embedding_session = MagicMock()
    fake_classifier_session = MagicMock()

    mock_encoding_result = MagicMock()
    mock_encoding_result.ids = [10, 20, 30, 40] 
    mock_encoding_result.attention_mask = [1, 1, 1, 1]
    fake_tokenizer.encode.return_value = mock_encoding_result

    fake_embeddings_output = np.array([[[0.5, 0.5, 0.5]]], dtype=np.float32)
    fake_embedding_session.run.return_value = [fake_embeddings_output]

    fake_logits = np.array([[0.1, 0.95]], dtype=np.float32)
    fake_classifier_session.run.return_value = [fake_logits]
    
    input_meta = MagicMock()
    input_meta.name = "dummy_input_name"
    fake_classifier_session.get_inputs.return_value = [input_meta]


    app.tokenizer = fake_tokenizer
    app.embedding_session = fake_embedding_session
    app.classifier_session = fake_classifier_session


def test_api_predict_positive_flow():
    test_payload = {"text": "MLOps with GitHub Actions is quite powerful."}
    
    response = client.post("/predict", json=test_payload)

    assert response.status_code == 200
    data = response.json()
    assert "sentiment" in data
    assert "confidence" in data
    assert data["sentiment"] == "positive"
    assert data["confidence"] > 0.9
    assert data["text"] == test_payload["text"]

@pytest.mark.parametrize("bad_input", [{"text": ""}, {}])
def test_api_validation_error(bad_input):
    response = client.post("/predict", json=bad_input)
    assert response.status_code == 422