# src/trainer.py
import mlflow
import mlflow.sklearn
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from xgboost import XGBClassifier
import numpy as np
import pandas as pd
from .config import model_config, data_config
from .models.dl_models import LSTMModel, ConvLSTM
from .data_prep_seq import create_spatial_sequences
from .utils import get_logger

logger = get_logger(__name__)

class WildfireTrainer:
    def __init__(self):
        mlflow.set_experiment(model_config.experiment_name)
        logger.info(f"MLflow experiment: {model_config.experiment_name}")

    def train_tabular(self, model_name: str, X_train, y_train, X_val, y_val):
        run_name = model_config.run_name_rf if "RandomForest" in model_name else model_config.run_name_xgb
        with mlflow.start_run(run_name=f"{model_name}_{run_name}"):
            if "RandomForest" in model_name:
                model = RandomForestClassifier(
                    n_estimators=model_config.n_estimators_rf,
                    max_depth=model_config.max_depth_rf,
                    random_state=model_config.random_state,
                    n_jobs=-1,
                    class_weight='balanced'
                )
            elif "XGBoost" in model_name:
                model = XGBClassifier(
                    n_estimators=model_config.n_estimators_xgb,
                    max_depth=model_config.max_depth_xgb,
                    learning_rate=model_config.learning_rate_xgb,
                    random_state=model_config.random_state,
                    eval_metric='logloss'
                )

            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            y_prob = model.predict_proba(X_val)[:, 1]

            metrics = {
                "f1": f1_score(y_val, y_pred),
                "recall": recall_score(y_val, y_pred),
                "auc": roc_auc_score(y_val, y_prob)
            }

            mlflow.log_metrics(metrics)
            mlflow.log_params(model.get_params())
            input_example = X_train[:5].values if hasattr(X_train, 'values') else X_train[:5]
            mlflow.sklearn.log_model(model, f"{model_name}_model", input_example=input_example)

            logger.info(f"{model_name} - F1: {metrics['f1']:.3f}, AUC: {metrics['auc']:.3f}")

    def train_dl(self, model_name: str, sample_ratio: float = 0.1):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Training {model_name} on {device} with sample_ratio={sample_ratio}")

        loader = create_spatial_sequences(sample_ratio=sample_ratio)
        run_name = model_config.run_name_lstm if "LSTM" in model_name else model_config.run_name_convlstm

        with mlflow.start_run(run_name=f"{model_name}_{run_name}"):
            if "LSTM" in model_name:
                input_size = len(data_config.features) * data_config.grid_size * data_config.grid_size
                model = LSTMModel(input_size).to(device)
                def preprocess(x):
                    b, t, c, h, w = x.shape
                    return x.reshape(b, t, c * h * w)
            else:
                model = ConvLSTM(in_c=len(data_config.features)).to(device)
                def preprocess(x): return x

            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            best_loss = float('inf')
            patience_counter = 0

            for epoch in range(model_config.epochs_dl):
                model.train()
                train_loss = 0
                for Xb, yb in loader:
                    Xb, yb = Xb.to(device), yb.to(device)
                    Xb = preprocess(Xb)
                    optimizer.zero_grad()
                    pred = model(Xb)
                    loss = criterion(pred, yb)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()

                avg_loss = train_loss / len(loader)
                mlflow.log_metric("train_loss", avg_loss, step=epoch)

                # === ACCURACY LOGGING ===
                with torch.no_grad():
                    acc = ((pred > 0.5) == yb).float().mean().item()
                    mlflow.log_metric("train_accuracy", acc, step=epoch)

                # === EARLY STOPPING ===
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    patience_counter = 0
                    torch.save(model.state_dict(), f"artifacts/best_{model_name}.pt")
                else:
                    patience_counter += 1
                    if patience_counter >= model_config.patience:
                        logger.info(f"Early stopping at epoch {epoch}")
                        break

            mlflow.pytorch.log_model(model, f"{model_name}_model")
            logger.info(f"{model_name} - Training complete")