import numpy as np
import sys
import os
import pickle
import pandas as pd
import json
from src.utils.main_utils import MainUtils
from src.logger import logging
from src.exception import CustomException

def extract_top_save_features(
    preprocessor_path="artifacts/preprocessor.pkl",
    model_path="artifacts/model.pkl",
    train_csv_path="artifacts/train_processed.csv",
    output_json_path="artifacts/top_features.json",
    top_n=6  # Changed to match your 6 core features
):
    try:
        logging.info("Loading preprocessor and model objects.")
        preprocessor = MainUtils().load_object(preprocessor_path)
        model = MainUtils().load_object(model_path)
        train_df = pd.read_csv(train_csv_path)

        # Get only the core features we're actually using
        feature_names = preprocessor.get('feature_names', 
                                      ['Age', 'NumOfProducts', 'IsActiveMember', 
                                       'Balance', 'CreditScore', 'EstimatedSalary'])
        
        # For models that support feature_importances_
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            # Ensure we don't try to get more features than exist
            top_n = min(top_n, len(feature_names))
            top_indices = np.argsort(importances)[::-1][:top_n]
            top_features = [feature_names[i] for i in top_indices]
        else:
            # For models without feature importance (like logistic regression)
            top_features = feature_names[:top_n]
        
        logging.info(f"Using features: {top_features}")

        # Calculate means only for the features we're using
        feature_means = train_df[top_features].mean().to_dict()

        # Save to JSON
        with open(output_json_path, "w") as f:
            json.dump({
                "top_features": top_features, 
                "feature_means": feature_means
            }, f, indent=4)

        logging.info(f"Features and means saved to: {output_json_path}")
        return top_features, feature_means
    
    except Exception as e:
        logging.error("Error occurred while extracting and saving top features.")
        raise CustomException(e, sys)