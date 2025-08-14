import os

# Path to your churn dataset
CHURN_DATA_PATH = os.path.join("data", "churn.csv")

# Artifact folder for storing models and outputs
artifact_folder = "artifacts"

# Model file details
MODEL_FILE_NAME = "model.pkl"
MODEL_FILE_EXTENSION = ".pkl"

# Target column in your churn dataset
TARGET_COLUMN = "Churn"  # change if your churn dataset has a different target column
