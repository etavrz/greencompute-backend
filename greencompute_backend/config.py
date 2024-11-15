import os

from dotenv import load_dotenv

load_dotenv()

ENVIRON = os.getenv("ENVIRON", "dev")
ROOT_PATH = "/api" if ENVIRON == "prod" else ""
AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET")

# Model and data files
MODELS = {
    "carbon-emissions": "xgb_carbon_model.pkl",
    "it-electricity": "gbr_it_electricity_model.pkl",
    "active-idle": "rf_activeidle_model.pkl",
    "pue": "xgb_pue_sklearn.pkl",
}
CARBON_MODEL = "xgb_carbon_model.pkl"

DATA_FILE = "cloud_embodied_emissions.csv"
