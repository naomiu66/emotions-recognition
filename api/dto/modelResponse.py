from pydantic import BaseModel

class ModelResponse(BaseModel):
    label: str
    confidence: float