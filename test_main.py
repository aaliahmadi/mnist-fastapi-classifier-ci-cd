from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_predict_endpoint():
    # This is a basic test to check if the endpoint is reachable.
    # You would need to provide a valid image file for a full test.
    response = client.post("/predict")
    assert response.status_code != 404 # The endpoint exists
    # For a proper test, you'd send a sample image and check the prediction.
    print("Test passed: /predict endpoint is reachable.")
