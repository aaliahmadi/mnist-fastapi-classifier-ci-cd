import io
import os
import torch
import torch.nn as nn
import torch.optim as optim
from fastapi import FastAPI, File, UploadFile
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image

app = FastAPI()

# --- Config ---
batch_size = 64
epochs = 25
lr = 0.001
model_path = "mnist_simple.pth"

# --- Model ---
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# --- Transforms ---
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# --- Training function ---
def train_and_save_model():
    print("Training model...")
    train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = SimpleNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    return model

# --- Startup: train if needed ---
if os.path.exists(model_path):
    model = SimpleNet()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    print("Loaded existing model.")
else:
    model = train_and_save_model()

model.eval()

# --- Routes ---
@app.get("/")
def home():
    return {"message": "MNIST classifier ready. Use /predict to classify an image."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("L")
    img_tensor = transform(image).unsqueeze(0)  # [1, 1, 28, 28]

    with torch.no_grad():
        outputs = model(img_tensor)
        predicted = outputs.argmax(dim=1).item()

    return {"predicted_digit": predicted}
