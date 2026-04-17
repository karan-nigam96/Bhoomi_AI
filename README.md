# Bhoomi AI

Bhoomi AI is a Flask-based crop recommendation and fertilizer guidance application.
It includes multiple crop prediction pipelines, a fertilizer recommendation flow, and a web UI for farmers and agronomy workflows.

## Features

- Crop recommendation web app built with Flask
- Multiple model versions for crop prediction
- Fertilizer recommendation support
- Zone and district lookup helpers
- REST API endpoints for predictions and soil defaults
- Static web UI with HTML templates, CSS, and JavaScript

## Project Structure

- `app.py` - Flask application entry point
- `functions.py` - crop prediction helpers and model loaders
- `fertilizer_functions.py` - fertilizer recommendation helpers
- `train_v2.py`, `train_v3.py`, `feature_engineering.py` - training and dataset generation scripts
- `dataset/` - training and support datasets
- `models/` - saved model artifacts
- `templates/` - HTML pages
- `static/` - CSS and JavaScript assets

## Requirements

- Python 3.10+
- Packages listed in `requirements.txt`

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the App

Start the Flask app directly:

```bash
python app.py
```

Or use the helper startup script:

```bash
python start_server.py
```

The app runs on `http://127.0.0.1:5000` by default.

## Main Routes

- `GET /` - home page
- `GET, POST /predict` - crop prediction page and form submission
- `GET /crops` - crop information page
- `GET /fertilizers` - fertilizer guide page
- `GET /fertilizer-recommend` - fertilizer recommendation page
- `POST /api/fertilizer/recommend` - fertilizer API
- `GET /about` - project information
- `GET /api/get_districts/<state>` - district lookup API
- `GET /api/soil_defaults/<zone_id>` - soil defaults API
- `POST /api/predict` - crop prediction API
- `POST /api/predict_legacy` - legacy crop prediction API

## Notes

- Model and dataset files are included in the repository so the app can run without retraining.
- Generated report files and temporary documentation were removed to keep the repo focused on runnable code.