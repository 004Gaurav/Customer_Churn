import pickle
import json

# Load artifacts
with open("artifacts/model.pkl", "rb") as f:
    model = pickle.load(f)
with open("artifacts/preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

print("Model expects these features:", model.feature_names_in_)
print("Preprocessor expects:", preprocessor.get('feature_names', 'Not found'))


# python  check_features.py
