from fastapi import FastAPI, HTTPException
from inference.schemas import WildfireRequest, WildfireResponse
from inference.model_loader import WildfireModel

import pandas as pd
from pathlib import Path

app = FastAPI(
    title="Wildfire Risk Inference API",
    version="1.0.0",
    description="Production inference service for wildfire risk prediction"
)

model = WildfireModel()

# -----------------------------
# PATH FOR DRIFT MONITORING
# -----------------------------
CURRENT_DIR = Path("monitoring/current")
CURRENT_DIR.mkdir(parents=True, exist_ok=True)
CURRENT_PATH = CURRENT_DIR / "current.parquet"


@app.get("/")
def health():
    return {"status": "healthy"}


@app.post("/predict", response_model=WildfireResponse)
def predict(req: WildfireRequest):
    try:
        # features come as dict (correct)
        features_dict = req.features

        prob, pred = model.predict(features_dict)

        # -----------------------------
        # SAVE LIVE DATA FOR EVIDENTLY
        # -----------------------------
        row = {
            **features_dict,
            "prediction": pred,
            "probability": prob,
        }

        new_df = pd.DataFrame([row])

        if CURRENT_PATH.exists():
            old_df = pd.read_parquet(CURRENT_PATH)
            new_df = pd.concat([old_df, new_df], ignore_index=True)

        new_df.to_parquet(CURRENT_PATH, index=False)

        return WildfireResponse(
            wildfire_probability=prob,
            wildfire_prediction=pred
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
