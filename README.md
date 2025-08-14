# Customer Churn Prediction

This project is a Flask-based web application for predicting customer churn using a machine learning model. It allows users to input customer data and receive predictions on whether a customer is likely to churn.

## Features

- Web interface for customer data input
- Machine learning model for churn prediction
- Configurable model and preprocessor paths
- Easy deployment and local development

## Project Structure

```
.
├── app.py
├── artifacts/
│   ├── model.pkl
│   ├── preprocessor.pkl
│   └── top_features.json
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.7+
- pip

### Installation

1. **Clone the repository:**
   ```sh
   git clone https://github.com/004Gaurav/Customer_Churn.git
   cd Customer_Churn
   ```

2. **Create a virtual environment and activate it:**
   ```sh
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

4. **Configure environment variables:**
   - Edit the `.env` file as needed.

5. **Add model artifacts:**
   - Place your `model.pkl`, `preprocessor.pkl`, and `top_features.json` in the `artifacts/` directory.

### Running the App

```sh
flask run
```

The app will be available at [http://localhost:5000](http://localhost:5000).

## Configuration

- All configuration is managed via the `.env` file.
- Model and preprocessor paths can be changed as needed.

## License

This project is licensed under the MIT License.

