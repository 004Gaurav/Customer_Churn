import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException
from src.utils.main_utils import MainUtils
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


@dataclass
class DataTransformationConfig:
    artifact_dir: str = os.path.join("artifacts")
    train_file_path: str = os.path.join(artifact_dir, "train.csv")
    test_file_path: str = os.path.join(artifact_dir, "test.csv")
    transformed_train_file_path: str = os.path.join(artifact_dir, "train_processed.csv")
    transformed_test_file_path: str = os.path.join(artifact_dir, "test_processed.csv")
    transformed_object_file_path: str = os.path.join(artifact_dir, "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()
        self.utils = MainUtils()

    def initiate_data_transformation(self):
        logging.info("Data Transformation started")
        try:
            # Load train and test data
            train_df = pd.read_csv(self.config.train_file_path)
            test_df = pd.read_csv(self.config.test_file_path)
            logging.info(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

            # Split features and target
            X_train = train_df.drop(columns=["Churn"])
            y_train = train_df["Churn"]
            X_test = test_df.drop(columns=["Churn"])
            y_test = test_df["Churn"]

            # Separate categorical and numerical columns
            categorical_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]
            numerical_cols = [col for col in X_train.columns if X_train[col].dtype in ["int64", "float64"]]
            logging.info(f"Categorical columns: {categorical_cols}")
            logging.info(f"Numerical columns: {numerical_cols}")

            # Numerical pipeline
            numeric_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler())
            ])

            X_train_num = numeric_pipeline.fit_transform(X_train[numerical_cols])
            X_test_num = numeric_pipeline.transform(X_test[numerical_cols])
            logging.info("Numerical features transformed")

            # Categorical encoding
            X_train_cat = pd.get_dummies(X_train[categorical_cols], drop_first=True)
            X_test_cat = pd.get_dummies(X_test[categorical_cols], drop_first=True)

            # Align train and test categorical columns
            X_train_cat, X_test_cat = X_train_cat.align(X_test_cat, join="left", axis=1, fill_value=0)
            logging.info("Categorical features encoded and aligned")

            # Combine numerical and categorical
            X_train_processed = np.hstack([X_train_num, X_train_cat])
            X_test_processed = np.hstack([X_test_num, X_test_cat])

            # Save processed CSVs
            all_feature_names = numerical_cols + list(X_train_cat.columns)
            train_out = pd.DataFrame(X_train_processed, columns=all_feature_names)
            test_out = pd.DataFrame(X_test_processed, columns=all_feature_names)
            train_out["Churn"] = y_train.values
            test_out["Churn"] = y_test.values

            train_out.to_csv(self.config.transformed_train_file_path, index=False)
            test_out.to_csv(self.config.transformed_test_file_path, index=False)

            # Save preprocessor object
            preprocessor = {
                "numeric_cols": numerical_cols,
                "categorical_cols": categorical_cols,
                "numeric_pipeline": numeric_pipeline,
                "encoded_columns": list(X_train_cat.columns),
                "all_feature_names": numerical_cols + list(X_train_cat.columns)  # actual training features
            }
            self.utils.save_object(self.config.transformed_object_file_path, preprocessor)

            logging.info("Data Transformation completed successfully")

            return (
                self.config.transformed_train_file_path,
                self.config.transformed_test_file_path,
                self.config.transformed_object_file_path
            )

        except Exception as e:
            logging.error(f"Error during data transformation: {str(e)}")
            raise CustomException(e, sys)
        
        # 
