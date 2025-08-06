import gspread
from oauth2client.service_account import ServiceAccountCredentials

# Setup auth
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
client = gspread.authorize(creds)

# Connect to sheet
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
    records = sheet.get_all_records()
    return records
