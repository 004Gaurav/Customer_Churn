import os
import sys
import json
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
from src.constant import TARGET_COLUMN
from src.utils.main_utils import MainUtils


class PredictPipeline:
    """
    A pipeline for loading saved model artifacts, preprocessing input data, 
    and generating predictions.
    """

    def __init__(self):
        self.utils = MainUtils()
        self.model_path = os.path.join("artifacts", "model.pkl")
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
        self.top_features_path = os.path.join("artifacts", "top_features.json")

    def load_artifacts(self):
        """
        Loads the model, preprocessor, and feature details from disk.
        """
        try:
            logging.info("Loading saved artifacts...")

            model = self.utils.load_object(self.model_path)
            preprocessor = self.utils.load_object(self.preprocessor_path)

            with open(self.top_features_path, "r") as f:
                feature_info = json.load(f)

            top_features = feature_info.get("top_features", [])
            feature_means = feature_info.get("feature_means", {})

            logging.info("Artifacts loaded successfully.")
            return model, preprocessor, top_features, feature_means

        except Exception as e:
            logging.error("Error while loading artifacts.")
            raise CustomException(e, sys)

    def preprocess_input(self, input_df, preprocessor, feature_means):
        """
        Prepares input DataFrame for prediction using the saved preprocessing pipeline.
        """
        try:
            logging.info("Preprocessing input data...")

            # Drop unnecessary columns if present
            drop_cols = ["SK_ID_CURR"]
            input_df = input_df.drop(columns=[col for col in drop_cols if col in input_df.columns], errors="ignore")

            # Ensure all required features are present
            all_features = preprocessor["numeric_cols"] + preprocessor["categorical_cols"]
            input_df = input_df.reindex(columns=all_features, fill_value=np.nan)
            input_df = input_df.fillna(feature_means)

            # Transform numeric features
            numeric_cols = preprocessor["numeric_cols"]
            categorical_cols = preprocessor["categorical_cols"]

            X_num = preprocessor["numeric_pipeline"].transform(input_df[numeric_cols])

            # Transform categorical features (One-hot encoding)
            X_cat = pd.get_dummies(input_df[categorical_cols], drop_first=True)
            X_cat = X_cat.reindex(columns=preprocessor["encoded_columns"], fill_value=0)

            # Combine numeric and categorical features
            X_processed = np.hstack([X_num, X_cat.values])

           
            logging.info("Preprocessing completed.")
            return X_processed

        except Exception as e:
            logging.error("Error during input preprocessing.")
            raise CustomException(e, sys)

    def predict_from_csv(self, csv_path):
        """
        Predicts churn values from a CSV file and saves the results.
        """
        try:
            model, preprocessor, top_features, feature_means = self.load_artifacts()

            # Load CSV
            input_df = pd.read_csv(csv_path)
            if "Unnamed: 0" in input_df.columns:
                input_df = input_df.drop(columns="Unnamed: 0")

            # Preprocess and predict
            X_processed = self.preprocess_input(input_df, preprocessor, feature_means)
            preds = model.predict(X_processed)

            # Map prediction values
            target_mapping = {0: "bad", 1: "good"}
            input_df[TARGET_COLUMN] = pd.Series(preds).map(target_mapping)

            # Log prediction counts
            counts = input_df[TARGET_COLUMN].value_counts()
            logging.info(f"Prediction counts:\n{counts}")

            # Save output
            output_dir = os.path.join("artifacts", "predictions")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "prediction_file.csv")
            input_df.to_csv(output_path, index=False)

            logging.info(f"Predictions saved to {output_path}")
            return output_path

        except Exception as e:
            raise CustomException(e, sys)

    def predict_from_dict(self, user_input_dict):
        """
        Predicts churn value from a single dictionary of user inputs.
        """
        try:
            model, preprocessor, top_features, feature_means = self.load_artifacts()

            # Ensure all features exist in the input
            all_features = preprocessor["numeric_cols"] + preprocessor["categorical_cols"]
            full_input = {f: user_input_dict.get(f, feature_means.get(f, np.nan)) for f in all_features}

            # Convert to DataFrame and preprocess
            input_df = pd.DataFrame([full_input])
            X_processed = self.preprocess_input(input_df, preprocessor, feature_means)

            # Predict and return result
            pred = model.predict(X_processed)[0]
            return pred

        except Exception as e:
            raise CustomException(e, sys)
