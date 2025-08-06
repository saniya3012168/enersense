import os
import base64
import json
import gspread
from oauth2client.service_account import ServiceAccountCredentials

def load_credentials_from_env():
    encoded = os.environ.get("GOOGLE_CREDENTIALS_BASE64")
    if not encoded:
        raise ValueError("Missing GOOGLE_CREDENTIALS_BASE64 env variable")
    decoded = base64.b64decode(encoded).decode("utf-8")
    return json.loads(decoded)

scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds_dict = load_credentials_from_env()
creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
client = gspread.authorize(creds)

sheet = client.open("EnerSense Load Data").worksheet("PredictionLogs")

def log_prediction_to_sheet(data):
    sheet.append_row([
        data["timestamp"],
        data["temperature"],
        data["humidity"],
        data["solar"],
        data["appliances"],
        data["income"],
        data["predicted_kWh"],
        data.get("actual_kWh", "")
    ])

def get_predictions_from_sheet():
    return sheet.get_all_records()
