"""
Example command to start FastAPI app:
uvicorn main:app --reload --port 8000

The app will be hosted on localhost:8000 with this command.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
import asyncio
import json
import numpy as np

from model import DigitRecognizer
from inference import single_image_inference

configs = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup procedure for the app.

    args:
    - app (FastAPI): the app to start up
    """
    configs["model_path"] = "best_model.pth"    

    print("Application startup...")
    yield

    print("Shutdown: clearing global configurations...")
    configs.clear()
    print("Application shutdown...")


app = FastAPI(lifespan=lifespan)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class inferenceContent(BaseModel):
    """
    Parameter structure to be passed for prediction.
    """
    img_array: list[list[float]]


@app.post("/api/inference")
async def inference_pred(content: inferenceContent) -> str:
    """
    Provides inference prediction API to frontend.

    args:
    - content (inferenceContent): parameter class containing the image array to perform prediction on

    returns:
    - a JSON string containing the prediction results and probabilities
    """
    image = np.array(content.img_array)
    model_architecture = DigitRecognizer()

    try:
        predictions = single_image_inference(
            model_path=configs["model_path"],
            image_array=image,
            model_architecture=model_architecture
        )
    
    except:
        predictions = {}

    predictions_jsonstr = json.dumps(predictions)

    return predictions_jsonstr