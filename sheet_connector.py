import os
import json
import base64
import gspread
from oauth2client.service_account import ServiceAccountCredentials

def load_credentials_from_env():
    # Get the Base64 encoded JSON from environment variable
    encoded = os.environ.get("GOOGLE_CREDENTIALS_BASE64")
    if not encoded:
        raise ValueError("Missing GOOGLE_CREDENTIALS_BASE64 environment variable")

    # Decode Base64 back to JSON dictionary
    decoded = base64.b64decode(encoded).decode("utf-8")
    creds_dict = json.loads(decoded)
    return creds_dict

# Setup the scope and credentials
scope = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive"
]
credentials = ServiceAccountCredentials.from_json_keyfile_dict(load_credentials_from_env(), scope)
client = gspread.authorize(credentials)

# Connect to the Google Sheet and worksheet
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
        data.get("actual_kWh", "")  # optional field
    ])

def get_predictions_from_sheet():
    return sheet.get_all_records()

def get_prediction_summary():
    records = get_predictions_from_sheet()
    total_kwh = sum(float(r["predicted_kWh"]) for r in records)
    total_cost = round(total_kwh * 6.5, 2)  # ₹6.5 per kWh
    return round(total_kwh, 2), total_cost

def export_sheet_to_csv():
    records = get_predictions_from_sheet()
    if not records:
        return "timestamp,temperature,humidity,solar,appliances,income,predicted_kWh,actual_kWh\n"

    headers = records[0].keys()
    rows = [",".join(headers)]
    for r in records:
        rows.append(",".join(str(r[h]) for h in headers))
    return "\n".join(rows)
