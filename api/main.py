from transformers import BertForSequenceClassification, AutoTokenizer
import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv, dotenv_values
from api.dto.modelRequest import ModelRequest
from api.dto.modelResponse import ModelResponse
from api.services.predict import fit

load_dotenv()

logger = logging.getLogger(__name__)

root_path = os.getcwd()
env = os.getenv("ENVIRONMENT")
repo = os.getenv("MODEL_REPO_NAME")

if env == "DEV":
    model_path = os.path.join(root_path, "models")
else:
    model_path = repo

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = BertForSequenceClassification.from_pretrained(model_path)
    app.state.tokenizer = AutoTokenizer.from_pretrained(model_path)
    logger.log(level=0, msg="Service started")
    yield
    logger.info("Service shutting down...")

app = FastAPI(
    lifespan=lifespan,
    title="Emotions recognition service API",
    description="API for emotions recognition using fine tuned BERT model",
    version="1.0.0"
)

@app.post('/predict', tags=["Emotion Model"], response_model=ModelResponse)
async def predict(request: ModelRequest):
    model = app.state.model
    tokenizer = app.state.tokenizer
    try:
        output = fit(model=model, tokenizer=tokenizer, input=request.text)
    except Exception:
        logger.exception(msg="Model inference failed")
        raise HTTPException(status_code=500, detail="Model inference failed")
    
    return ModelResponse(label=output['label'], confidence=output['confidence'])