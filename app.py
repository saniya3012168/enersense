from flask import Flask, request, jsonify, render_template, send_file
from enersense import generate_synthetic_data, train_consumption_model, forecast_solar_generation, optimize_grid
from sheet_connector import (
    log_prediction_to_sheet, get_predictions_from_sheet,
    get_prediction_summary, export_sheet_to_csv
)
import datetime
import io

app = Flask(__name__)

# Train model on startup
data = generate_synthetic_data()
model = train_consumption_model(data)


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    form = request.form
    df = data.iloc[[0]].copy()
    df["Temperature"] = float(form['temperature'])
    df["Humidity"] = float(form['humidity'])
    df["Solar"] = float(form['solar'])
    df["Appliances"] = int(form['appliances'])
    df["Income"] = float(form['income'])

    prediction = model.predict(df[["Temperature", "Humidity", "Solar", "Appliances", "Income"]])[0]

    log_prediction_to_sheet({
        "timestamp": str(datetime.datetime.now()),
        "temperature": df["Temperature"].iloc[0],
        "humidity": df["Humidity"].iloc[0],
        "solar": df["Solar"].iloc[0],
        "appliances": df["Appliances"].iloc[0],
        "income": df["Income"].iloc[0],
        "predicted_kWh": round(prediction, 2)
    })

    return render_template("index.html", result=round(prediction, 2))


@app.route('/history')
def history():
    records = get_predictions_from_sheet()
    return render_template("dashboard.html", records=records)


@app.route('/summary')
def summary():
    total_kwh, total_cost = get_prediction_summary()
    return f"<h3>Total Predicted Energy: {total_kwh} kWh<br>Total Estimated Cost: ₹{total_cost}</h3>"


@app.route('/download')
def download():
    csv_data = export_sheet_to_csv()
    return send_file(io.BytesIO(csv_data.encode()), mimetype='text/csv',
                     download_name='prediction_logs.csv', as_attachment=True)


@app.route('/solar-now')
def solar_now():
    solar = forecast_solar_generation()[0]
    return render_template("solar_now.html", solar=round(solar, 2))


@app.route('/model-info')
def model_info():
    from sklearn.metrics import mean_squared_error, r2_score
    X = data[["Temperature", "Humidity", "Solar", "Appliances", "Income"]]
    y = data["Energy_kWh"]
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    return render_template("model_info.html", r2=round(r2, 3), mse=round(mse, 3))


@app.route('/compare')
def compare():
    records = get_predictions_from_sheet()
    timestamps = [r['timestamp'] for r in records]
    predictions = [float(r['predicted_kWh']) for r in records]
    actuals = [float(r.get('actual_kWh', 0)) for r in records]  # fallback to 0 if missing
    return render_template("compare_chart.html",
                           timestamps=timestamps,
                           predictions=predictions,
                           actuals=actuals)


@app.route('/charts')
def charts():
    records = get_predictions_from_sheet()
    timestamps = [r['timestamp'] for r in records]
    predictions = [float(r['predicted_kWh']) for r in records]
    return render_template("chart_dashboard.html",
                           timestamps=timestamps,
                           predictions=predictions)


from agent import run_energy_agent  # at the top

@app.route('/agent-decision', methods=['POST'])
def agent_decision():
    input_data = request.get_json()
    result = run_energy_agent(input_data)
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)
