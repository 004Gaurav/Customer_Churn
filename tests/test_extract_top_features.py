import sys
import os

# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.extract_top_features import extract_top_save_features


if __name__ == "__main__":
    print("==== Testing Extract Top Features ====")
    try:
        top_features, features_means = extract_top_save_features()

        print(" The top features are:", top_features)
        print(" Feature means are:", features_means)

        # Check if the JSON file was created
        json_path = os.path.join("artifacts", "top_features.json")
        if os.path.exists(json_path):
            print(f" top_features.json created at: {json_path}")
        else:
            print(" top_features.json file not found!")
    except Exception as e:
        print(f" Error during extract top features test: {e}")


# python tests/test_extract_top_features.py