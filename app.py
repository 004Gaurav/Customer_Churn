from flask import Flask, render_template, request, send_from_directory
from datetime import datetime
import pandas as pd
import numpy as np
import json
import os
from src.utils.main_utils import MainUtils
from src.pipeline.prediction_pipeline import PredictPipeline

# Initialize Flask app
app = Flask(__name__)

# ======================
# 1. TEMPLATE HELPERS
# ======================
@app.context_processor
def inject_template_vars():
    """Make these variables available in all templates"""
    return {
        'current_year': datetime.now().year,
        'app_name': 'Customer Churn Predictor'
    }

# ======================
# 2. LOAD MODEL ARTIFACTS
# ======================
def load_artifacts():
    """Load all required machine learning artifacts"""
    try:
        # Load feature configuration
        with open("artifacts/top_features.json") as f:
            feature_info = json.load(f)
        
        # Load ML model and preprocessor
        preprocessor = MainUtils().load_object("artifacts/preprocessor.pkl")
        model = MainUtils().load_object("artifacts/model.pkl")
        
        return {
            'top_features': feature_info["top_features"],
            'feature_means': feature_info["feature_means"],
            'preprocessor': preprocessor,
            'model': model,
            'all_features': preprocessor['numeric_cols'] + preprocessor['encoded_columns']
        }
    except Exception as e:
        print(f"🚨 Error loading artifacts: {str(e)}")
        raise

# Load artifacts once when starting the app
ml_data = load_artifacts()

# ======================
# 3. ROUTES
# ======================
@app.route('/', methods=['GET', 'POST'])
def home():
    """Main prediction page"""
    prediction = None
    error = None
    
    if request.method == 'POST':
        try:
            # 1. Get user input
            user_input = {f: float(request.form[f]) for f in ml_data['top_features']}
            
            # 2. Prepare complete feature set
            full_input = {f: user_input.get(f, ml_data['feature_means'][f]) 
                         for f in ml_data['all_features']}
            
            # 3. Convert to DataFrame
            input_df = pd.DataFrame([full_input])
            
            # 4. Process numerical features
            X_num = ml_data['preprocessor']['numeric_pipeline'].transform(
                input_df[ml_data['preprocessor']['numeric_cols']]
            )
            
            # 5. Process categorical features
            X_cat = pd.get_dummies(
                input_df[ml_data['preprocessor']['categorical_columns']], 
                drop_first=True
            )
            X_cat = X_cat.reindex(
                columns=ml_data['preprocessor']['categorical_columns'], 
                fill_value=0
            )
            
            # 6. Make prediction
            X_processed = np.hstack([X_num, X_cat.values])
            prediction = ml_data['model'].predict(X_processed)[0]
            
        except Exception as e:
            error = f"Prediction failed: {str(e)}"
            print(f"⚠️ {error}")

    return render_template(
        'form.html',
        top_features=ml_data['top_features'],
        prediction=prediction,
        error=error
    )

# ======================
# 4. RUN THE APP
# ======================
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))  # defaults to 10000 if PORT not set

    
    # For Windows development:
    try:
        from waitress import serve
        print(f"🚀 Server running on http://localhost:{port} (Waitress)")
        serve(app, host="0.0.0.0", port=port)
    
    # For production/Linux:
    except ImportError:
        print(f"🚀 Server running on http://localhost:{port} (Flask development server)")
        app.run(host="0.0.0.0", port=port)