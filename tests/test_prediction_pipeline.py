import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline.prediction_pipeline import PredictPipeline

if __name__ == "__main__":
    try:
        print("==== Testing Predict Pipeline (batch prediction) ====")

        # Provide correct CSV path
        test_csv_path = os.path.join('notebook', 'data', 'Churn.csv')
        if not os.path.exists(test_csv_path):
            raise FileNotFoundError(f"{test_csv_path} not found!")

        pipeline = PredictPipeline()
        output_path = pipeline.predict_from_csv(test_csv_path)

        print(f"✅ Prediction file created at: {output_path}")
        print("Predict pipeline test completed successfully.")

    except Exception as e:
        print(f"❌ Error during predict pipeline test: {e}")


# python tests/test_prediction_pipeline.py