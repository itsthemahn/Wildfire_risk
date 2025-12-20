from pydantic import BaseModel, Field
from typing import Dict

class WildfireRequest(BaseModel):
    features: Dict[str, float]

class WildfireResponse(BaseModel):
    wildfire_probability: float
    wildfire_prediction: int
