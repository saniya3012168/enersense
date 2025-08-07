from flask import Flask, request, jsonify, render_template, send_file
from enersense import generate_synthetic_data, train_consumption_model, forecast_solar_generation, optimize_grid
import datetime
import io
import csv

app = Flask(__name__)

# Train model on startup
data = generate_synthetic_data()
model = train_consumption_model(data)

# In-memory storage for predictions
prediction_logs = []

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

    # Log to in-memory list
    log_entry = {
        "timestamp": str(datetime.datetime.now()),
        "temperature": df["Temperature"].iloc[0],
        "humidity": df["Humidity"].iloc[0],
        "solar": df["Solar"].iloc[0],
        "appliances": df["Appliances"].iloc[0],
        "income": df["Income"].iloc[0],
        "predicted_kWh": round(prediction, 2),
        "actual_kWh": ""  # Optional for now
    }
    prediction_logs.append(log_entry)

    return render_template("index.html", result=round(prediction, 2))

@app.route('/history')
def history():
    return render_template("dashboard.html", records=prediction_logs)

@app.route('/summary')
def summary():
    total_kwh = sum(float(r["predicted_kWh"]) for r in prediction_logs)
    total_cost = round(total_kwh * 6.5, 2)
    return f"<h3>Total Predicted Energy: {round(total_kwh, 2)} kWh<br>Total Estimated Cost: ₹{total_cost}</h3>"

@app.route('/download')
def download():
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=[
        "timestamp", "temperature", "humidity", "solar", "appliances", "income", "predicted_kWh", "actual_kWh"
    ])
    writer.writeheader()
    writer.writerows(prediction_logs)
    output.seek(0)

    return send_file(io.BytesIO(output.getvalue().encode()), mimetype='text/csv',
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
    timestamps = [r['timestamp'] for r in prediction_logs]
    predictions = [float(r['predicted_kWh']) for r in prediction_logs]
    actuals = [float(r.get('actual_kWh') or 0) for r in prediction_logs]
    return render_template("compare_chart.html", timestamps=timestamps, predictions=predictions, actuals=actuals)

@app.route('/charts')
def charts():
    timestamps = [r['timestamp'] for r in prediction_logs]
    predictions = [float(r['predicted_kWh']) for r in prediction_logs]
    return render_template("chart_dashboard.html", timestamps=timestamps, predictions=predictions)

# Energy agent route
from agent import run_energy_agent

@app.route('/agent-decision', methods=['POST'])
def agent_decision():
    input_data = request.get_json()
    result = run_energy_agent(input_data)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
