# train.py
from src.data_prep import DataProcessor
from src.trainer import WildfireTrainer
from src.config import data_config  # ← CORRECT IMPORT
import pandas as pd

def main():
    print("Wildfire MLOps: Training ALL Models (RF, XGB, LSTM, ConvLSTM) - Nov 10, 2025\n")

    # === TABULAR PIPELINE ===
    proc = DataProcessor()
    proc.prepare_tabular()

    # === LOAD ONLY FEATURES (NO datetime, NO Wildfire) ===
    X_train = pd.read_parquet("Data/processed/train.parquet")[data_config.features]
    X_val   = pd.read_parquet("Data/processed/val.parquet")[data_config.features]
    y_train = pd.read_parquet("Data/processed/train.parquet")['Wildfire']
    y_val   = pd.read_parquet("Data/processed/val.parquet")['Wildfire']

    trainer = WildfireTrainer()
    trainer.train_tabular("RandomForest", X_train, y_train, X_val, y_val)
    trainer.train_tabular("XGBoost", X_train, y_train, X_val, y_val)

    # === DEEP LEARNING ===
    print("\nTraining Deep Learning Models...")
    trainer.train_dl("LSTM",sample_ratio=1.0)
    trainer.train_dl("ConvLSTM",sample_ratio=1.0)

    print("\nALL MODELS TRAINED! Run: python -m mlflow ui")

if __name__ == "__main__":
    main()