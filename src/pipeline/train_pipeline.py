import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.utils.extract_top_features import extract_top_save_features
from src.exception import CustomException
import numpy as np
import pandas as pd


class TrainingPipeline:

    def start_data_ingestion(self):
        try:
            data_ingestion = DataIngestion()
            feature_store_file_path = data_ingestion.initiate_data_ingestion()
            return feature_store_file_path
        except Exception as e:
            raise CustomException(e, sys)

    def start_data_transformation(self):
        try:
            data_transformation = DataTransformation()
            train_arr_path, test_arr_path, preprocessor_path = data_transformation.initiate_data_transformation()
            return train_arr_path, test_arr_path, preprocessor_path
        except Exception as e:
            raise CustomException(e, sys)

    def start_model_training(self, train_arr_path, test_arr_path):
        try:
            # Load CSV files
            train_df = pd.read_csv(train_arr_path)
            test_df = pd.read_csv(test_arr_path)

            # Assuming your target column is named 'target'
            X_train = train_df.drop("Churn", axis=1)
            y_train = train_df["Churn"]

            X_test = test_df.drop("Churn", axis=1)
            y_test = test_df["Churn"]

            # Select only first 1000 rows for faster training
            X_train, y_train = X_train[:1000], y_train[:1000]
            X_test, y_test = X_test[:1000], y_test[:1000]

            model_trainer = ModelTrainer()
            model_path = model_trainer.initiate_model_trainer(X_train, y_train, X_test, y_test)
            return model_path
        except Exception as e:
            raise CustomException(e, sys)

    def run_pipeline(self):
        try:
            print("Starting data ingestion...")
            self.start_data_ingestion()
            print("Starting data transformation...")
            train_arr_path, test_arr_path, preprocessor_path = self.start_data_transformation()
            print("Starting model training...")
            model_path = self.start_model_training(train_arr_path, test_arr_path)
            print("Extracting top features...")
            extract_top_save_features()
            print("Training pipeline completed. Model saved at:", model_path)
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    TrainingPipeline().run_pipeline()


#python src/pipeline/train_pipeline.py
# This will execute the training pipeline and save the model and top features.