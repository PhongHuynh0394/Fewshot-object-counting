from fastapi.testclient import TestClient
from server.main import app

client = TestClient(app)

def test_connection():
    response = client.get("/healthcheck")
    assert response.status_code == 200
    assert response.json() == {"status": "Healthy server"}