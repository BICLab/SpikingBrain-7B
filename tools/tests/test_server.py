import pytest
from fastapi.testclient import TestClient
from spikingbrain.server.app import create_app
from spikingbrain.config import SpikingBrainConfig


@pytest.fixture
def client():
    config = SpikingBrainConfig(
        vocab_size=200,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
    )
    app = create_app(config=config)
    return TestClient(app)


def test_health_endpoint(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["model"] == "spikingbrain-7b"


def test_models_endpoint(client):
    res = client.get("/v1/models")
    assert res.status_code == 200
    data = res.json()
    assert len(data["data"]) == 1
    assert data["data"][0]["id"] == "spikingbrain-7b"


def test_chat_completions_sync(client):
    payload = {
        "model": "spikingbrain-7b",
        "messages": [{"role": "user", "content": "Hello SpikingBrain!"}],
        "max_tokens": 8,
        "stream": False,
    }
    res = client.post("/v1/chat/completions", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "choices" in data
    assert len(data["choices"]) > 0
    assert "usage" in data
    assert "neuromorphic_telemetry" in data["usage"]
    # Check custom neuromorphic headers
    assert "X-Neuromorphic-Sparsity" in res.headers
    assert "X-Energy-Savings-Ratio" in res.headers


def test_chat_completions_stream(client):
    payload = {
        "model": "spikingbrain-7b",
        "messages": [{"role": "user", "content": "Stream tokens"}],
        "max_tokens": 6,
        "stream": True,
    }
    res = client.post("/v1/chat/completions", json=payload)
    assert res.status_code == 200
    assert "text/event-stream" in res.headers["content-type"]
    lines = [line for line in res.text.split("\n") if line.startswith("data:")]
    assert len(lines) > 2
    assert lines[-1] == "data: [DONE]"
