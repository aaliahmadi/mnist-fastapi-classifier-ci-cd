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


2. Create a virtual environment (optional but recommended):

```
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows
```

3. Install dependencies:

```
pip install -r requirements.txt
```

## Usage

1. Run the FastAPI server:

```bash
uvicorn main:app --reload


2. Access the API:

- Home page: http://127.0.0.1:8000/

- Predict a digit:

```bash

curl -X POST "http://127.0.0.1:8000/predict" -F "file=@path_to_your_digit_image.png"


Response example:

{
  "predicted_digit": 7
}
