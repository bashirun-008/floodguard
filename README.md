# FloodGuard India

ML-powered district-level flood risk prediction and early warning dashboard built with Streamlit and XGBoost.

## Features
- National overview of high, moderate and low risk districts
- District-level risk map
- 5-day flood risk forecast with optional live weather (OpenWeatherMap)
- Automated alert panel
- Model insights and feature importance
- What-if policy simulator

## Run locally
    pip install -r requirements.txt
    streamlit run app.py

Optional live weather: set the environment variable OWM_API_KEY to your OpenWeatherMap key.

## Deployment
Deployed on AWS App Runner from this repository using apprunner.yaml.

## Data
Model trained on the Kaggle flood prediction dataset (naiyakhalid/flood-prediction-dataset). District values in the demo are simulated.
