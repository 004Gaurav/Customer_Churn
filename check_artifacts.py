import pickle
import json
from src.utils.main_utils import MainUtils

def check_artifacts():
    try:
        # Load model
        model = MainUtils().load_object("artifacts/model.pkl")
        print("Model expects:", model.feature_names_in_)
        
        # Load preprocessor
        preprocessor = MainUtils().load_object("artifacts/preprocessor.pkl")
        print("Preprocessor expects:", preprocessor.get("feature_names", "Not found"))
        
        # Load top features
        with open("artifacts/top_features.json") as f:
            top_features = json.load(f)
        print("top_features.json contains:", top_features["top_features"])
        
        # Verify alignment
        all_features = set(model.feature_names_in_).union(
            set(preprocessor.get("feature_names", [])).union(
            set(top_features["top_features"])))
            
        if len(all_features) > 6:
            print("\nWARNING: Feature mismatch detected!")
            print("These extra features appear in some artifacts:", all_features - {
                'Age', 'NumOfProducts', 'IsActiveMember', 
                'Balance', 'CreditScore', 'EstimatedSalary'
            })
        else:
            print("\nAll artifacts aligned on 6 core features")
            
    except Exception as e:
        print("Error checking artifacts:", str(e))

if __name__ == "__main__":
    check_artifacts()

    # python check_artifacts.py