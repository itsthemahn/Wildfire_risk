from fastapi import FastAPI, HTTPException
from inference.schemas import WildfireRequest, WildfireResponse
from inference.model_loader import WildfireModel

app = FastAPI(
    title="Wildfire Risk Inference API",
    version="1.0.0",
    description="Production inference service for wildfire risk prediction"
)

model = WildfireModel()

@app.get("/")
def health():
    return {"status": "healthy"}

@app.post("/predict", response_model=WildfireResponse)
def predict(req: WildfireRequest):
    try:
        prob, pred = model.predict(req.features)
        return WildfireResponse(
            wildfire_probability=prob,
            wildfire_prediction=pred
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
