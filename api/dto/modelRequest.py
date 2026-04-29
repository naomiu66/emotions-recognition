from pydantic import BaseModel, field_validator

class ModelRequest(BaseModel):
    text: str
    
    @field_validator("text")
    def validate_text(cls, v):
        if not v or not v.strip():
            raise ValueError("text must not be empty")
        return v