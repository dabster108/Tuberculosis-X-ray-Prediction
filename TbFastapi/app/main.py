from fastapi import FastAPI, UploadFile, File, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from PIL import Image
import io
import torch
from fastapi.staticfiles import StaticFiles



from app.model import load_model
from app.utils import predict_image
from app.schemas import PredictionResponse

app = FastAPI()

# Mount static directory for CSS or JS if needed
app.mount("/static", StaticFiles(directory="app/static"), name="static")

templates = Jinja2Templates(directory="app/templates")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "/Users/dikshanta/Downloads/TuberChestPrediction/tuber_model.pth"
model = load_model(MODEL_PATH, device)

# HTML form for uploading image
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})

# Endpoint for handling image upload and prediction
@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
    except Exception:
        return JSONResponse(status_code=400, content={"detail": "Invalid image file"})
    
    prediction = predict_image(image, model, device)
    return PredictionResponse(prediction=prediction)


# uvicorn app.main:app --reload
#uvicorn app.main:app --reload --port 8001