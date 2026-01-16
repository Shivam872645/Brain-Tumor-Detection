from flask import Flask, render_template, request
import torch
import torch.nn as nn
import cv2
import numpy as np
import os

# -------------------
# Flask App
# -------------------
app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# -------------------
# Load Model
# -------------------
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 30 * 30, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)

model = CNN()
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

# -------------------
# Image Preprocessing
# -------------------
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = np.transpose(img, (2, 0, 1))   # C,H,W
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
    return img

# -------------------
# Routes
# -------------------
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    image_path = None

    if request.method == "POST":
        file = request.files["image"]
        if file:
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(image_path)

            img = preprocess_image(image_path)
            output = model(img)
            prediction = "Tumor Detected" if output.item() > 0.5 else "Healthy Brain"

    return render_template("index.html",
                           prediction=prediction,
                           image_path=image_path)

# -------------------
# Run App
# -------------------
if __name__ == "__main__":
    app.run(debug=True)
