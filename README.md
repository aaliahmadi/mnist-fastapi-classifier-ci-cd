# MNIST FastAPI Classifier

A simple **FastAPI** application that classifies handwritten digits using a trained **PyTorch** model.  
You can upload an image of a digit (28x28 grayscale or RGB) and get its predicted class.

---

## Features
- Trains a simple fully connected neural network (`SimpleNet`) on the MNIST dataset.
- Serves the trained model through a REST API using FastAPI.
- Provides endpoints:
  - `GET /` → Home message.
  - `POST /predict` → Upload an image and get predicted digit.

---

## Project Structure
```
mnist-fastapi-classifier/
│
├── main.py # FastAPI application
├── mnist_simple.pth # Trained PyTorch model
├── requirements.txt # Python dependencies
├── README.md # This file
└── .gitignore # Git ignore for temporary files
```

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/AAliAhmadi/mnist-fastapi-classifier.git
cd mnist-fastapi-classifier
```

2. Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Run the FastAPI server:

```bash
uvicorn main:app --reload
```

2. Access the API:

- Home page: http://127.0.0.1:8000/

- Predict a digit:

```bash

curl -X POST "http://127.0.0.1:8000/predict" -F "file=@path_to_your_digit_image.png"
```

Response example:

{
  "predicted_digit": 7
}

---

## CI/CD Pipeline

This project includes a GitHub Actions workflow that automates testing, building, and deployment.

### Workflow Steps
1. **Continuous Integration (CI)**: Runs on every `push` or `pull_request` to the `main` branch.
   - Sets up Python 3.10.
   - Installs dependencies and runs tests using `pytest`.
2. **Continuous Deployment (CD)**: Runs after successful tests.
   - Builds a Docker image of the FastAPI application.
   - Publishes the image to **GitHub Container Registry (ghcr.io)**.

### How to Use
1. The workflow is defined in `.github/workflows/ci-cd.yml`.
2. The Docker image is built using the `Dockerfile` in the project root.
3. The image is tagged as `ghcr.io/aaliahmadi/mnist-fastapi-classifier:latest`.

You can find the published package under the **Packages** section of this repository.

For more details, check the workflow file and the Dockerfile in the repository.

---
