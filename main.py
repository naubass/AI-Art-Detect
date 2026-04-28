import torch
from torchvision import models, transforms
from PIL import Image
import io
import os
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Konfigurasi Model
MODEL_PATH = 'model_art_ai_human.pth'
device = torch.device('cpu')

# Load Model (Logika yang sama dengan bot kamu)
checkpoint = torch.load(MODEL_PATH, map_location=device)
class_names = checkpoint['class_names']
model = models.resnet50(weights=None)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, len(class_names))
model.load_state_dict(checkpoint['model_state'])
model.eval()

# Transformasi
img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Preprocess & Predict
        tensor = img_transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            score, preds = torch.max(probabilities, 1)

        label = class_names[preds[0]]
        confidence = score.item() * 100
        
        is_ai = "ai" in label.lower()
        
        return {
            "success": True,
            "label": "AI GENERATED" if is_ai else "HUMAN ART",
            "confidence": f"{confidence:.1f}%",
            "description": "Terdeteksi pola artifisial." if is_ai else "Terdeteksi goresan tangan manusia.",
            "color": "text-red-500" if is_ai else "text-green-500"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)