import sys
import os
import pandas as pd

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.components.model_trainer import ModelTrainer
from src.logger import logging
from src.exception import CustomException

if __name__ == "__main__":
    try:
        logging.info("==== Starting Model Training ====")
        print("Starting model training process...")

        # Load transformed train/test datasets from CSV
        train_df = pd.read_csv("artifacts/train_processed.csv")
        test_df = pd.read_csv("artifacts/test_processed.csv")

        # Separate features and target
        X_train = train_df.drop(columns=["Churn"])
        y_train = train_df["Churn"]

        X_test = test_df.drop(columns=["Churn"])
        y_test = test_df["Churn"]

        # Optional: Limit dataset for faster testing
        X_train, y_train = X_train.iloc[:1000], y_train.iloc[:1000]
        X_test, y_test = X_test.iloc[:200], y_test.iloc[:200]

        logging.info(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")

        # Train model
        trainer = ModelTrainer()
        model_path = trainer.initiate_model_trainer(X_train, y_train, X_test, y_test)

        logging.info(f"Model trained and saved at: {model_path}")
        print(f"✅ Model trained and saved at: {model_path}")

    except Exception as e:
        logging.error(f"Error during model training: {str(e)}")
        raise CustomException(e, sys)

# python tests/test_model_trainer.py