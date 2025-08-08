from flask import Flask, request, jsonify, render_template, send_file
from enersense import (
    generate_synthetic_data,
    train_consumption_model,
    forecast_solar_generation
)
from agent import run_energy_agent
import datetime
import io
import csv
import pandas as pd
import os
import logging

# --- Create Flask app first ---
app = Flask(__name__)
app.logger.setLevel(logging.INFO)

# Generate data and train model
data = generate_synthetic_data()
model = train_consumption_model(data)

# In-memory storage for predictions
prediction_logs = []

# Path for community CSV
COMMUNITIES_CSV = os.path.join("data", "communities.csv")

# ---------------- Utility: load community data ----------------
def load_communities():
    if not os.path.exists(COMMUNITIES_CSV):
        default = pd.DataFrame([
            {"community": "NorthTown", "population": 1200, "priority_score": 0.7},
            {"community": "EastVille", "population": 500, "priority_score": 0.9},
            {"community": "SouthPark", "population": 900, "priority_score": 0.6},
            {"community": "WestSide", "population": 400, "priority_score": 0.8}
        ])
        os.makedirs(os.path.dirname(COMMUNITIES_CSV), exist_ok=True)
        default.to_csv(COMMUNITIES_CSV, index=False)
    return pd.read_csv(COMMUNITIES_CSV)

# ---------------- Grid optimization logic ----------------
def optimize_grid(total_demand_kwh, available_solar_kwh, storage_capacity_kwh=10, storage_current_kwh=2, cost_grid=0.15, cost_renewable=0.05, loss_factor=0.02):
    alloc_renewable = min(available_solar_kwh, total_demand_kwh)
    remaining = total_demand_kwh - alloc_renewable
    use_storage = min(storage_current_kwh, remaining)
    remaining -= use_storage
    from_grid = max(0, remaining)
    loss = from_grid * loss_factor
    delivered_from_grid = from_grid - loss
    total_cost = round(cost_renewable * alloc_renewable + cost_grid * from_grid, 3)

    recommendations = []
    if alloc_renewable < total_demand_kwh:
        recommendations.append("Shift heavy loads to daylight hours when solar is available.")
    if storage_current_kwh < storage_capacity_kwh:
        recommendations.append("Consider increasing battery storage to store excess solar.")
    if loss > 0.5:
        recommendations.append("Investigate distribution losses (line upgrades or microgrids).")

    return {
        "demand_kWh": round(total_demand_kwh, 3),
        "renewable_used_kWh": round(alloc_renewable, 3),
        "storage_used_kWh": round(use_storage, 3),
        "grid_used_kWh": round(from_grid, 3),
        "grid_loss_kWh": round(loss, 3),
        "delivered_from_grid_kWh": round(delivered_from_grid, 3),
        "total_cost_$": total_cost,
        "recommendations": recommendations
    }

# ---------------- Equity allocation logic ----------------
def allocate_equity(total_available_kwh, communities_df):
    pop_norm = (communities_df["population"] - communities_df["population"].min()) / (
        communities_df["population"].max() - communities_df["population"].min() + 1e-9)
    weights = communities_df["priority_score"] * (1 + pop_norm)
    weights = weights.clip(lower=0)
    weights_sum = weights.sum() if weights.sum() > 0 else 1.0
    allocation = (weights / weights_sum) * total_available_kwh
    result = communities_df.copy()
    result["allocated_kWh"] = allocation.round(3)
    return result

# ---------------- Clean energy recommendation logic ----------------
def recommend_clean_energy(latest_log=None):
    recs = []
    if latest_log is None and prediction_logs:
        latest_log = prediction_logs[-1]
    if latest_log:
        if latest_log["predicted_kWh"] > 10:
            recs.append("High consumption — consider adding rooftop solar (3-5 kW) and battery storage.")
        if latest_log["appliances"] >= 6:
            recs.append("Consider smart plugs and load scheduling.")
        if latest_log["income"] and latest_log["income"] < 40000:
            recs.append("Explore community solar / subsidy programs.")
        if latest_log["solar"] > 3:
            recs.append("Good solar potential — upgrade inverter to increase yield.")
    else:
        recs.append("No data yet — make a prediction for personalized tips.")
    recs.extend([
        "Use energy-efficient appliances.",
        "Shift washing/dishwashing to midday if you have solar.",
        "Install programmable thermostats to cut peak loads."
    ])
    return list(dict.fromkeys(recs))  # remove duplicates

# ---------------- Routes ----------------
@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    form = request.form
    df = data.iloc[[0]].copy()
    df["Temperature"] = float(form.get('temperature', df["Temperature"].iloc[0]))
    df["Humidity"] = float(form.get('humidity', df["Humidity"].iloc[0]))
    df["Solar"] = float(form.get('solar', df["Solar"].iloc[0]))
    df["Appliances"] = int(form.get('appliances', df["Appliances"].iloc[0]))
    df["Income"] = float(form.get('income', df["Income"].iloc[0]))

    prediction = model.predict(df[["Temperature", "Humidity", "Solar", "Appliances", "Income"]])[0]
    log = {
        "timestamp": str(datetime.datetime.now()),
        "temperature": round(df["Temperature"].iloc[0], 3),
        "humidity": round(df["Humidity"].iloc[0], 3),
        "solar": round(df["Solar"].iloc[0], 3),
        "appliances": int(df["Appliances"].iloc[0]),
        "income": round(df["Income"].iloc[0], 2),
        "predicted_kWh": round(prediction, 2),
        "actual_kWh": ""
    }
    prediction_logs.append(log)
    return render_template("index.html", result=round(prediction, 2))

@app.route('/history')
def history():
    return render_template("dashboard.html", records=prediction_logs)

'''@app.route('/grid-opt')
def grid_opt():
    if not prediction_logs:
        return render_template("grid_optimization.html", optimized=None, msg="No prediction available.")
    latest = prediction_logs[-1]
    demand = float(latest["predicted_kWh"])
    solar_kwh = forecast_solar_generation(latest["solar"])
    result = optimize_grid(demand, solar_kwh, storage_capacity_kwh=12, storage_current_kwh=3)
    return render_template("grid_optimization.html", optimized=result, solar_kwh=round(solar_kwh, 3), demand=demand)'''

@app.route('/equity')
def equity():
    if not prediction_logs:
        return render_template("equity_dashboard.html", allocations=None, msg="No prediction history.")
    recent = prediction_logs[-10:] if len(prediction_logs) >= 10 else prediction_logs
    available = sum([r["solar"] * 0.15 * 10 / 1000 for r in recent])
    communities = load_communities()
    allocation_df = allocate_equity(available, communities)
    allocations = allocation_df.to_dict(orient="records")
    return render_template("equity_dashboard.html", allocations=allocations, available=round(available, 3))

@app.route('/recommendations')
def clean_energy():
    latest = prediction_logs[-1] if prediction_logs else None
    recs = recommend_clean_energy(latest)
    return render_template("clean_energy.html", recommendations=recs, latest=latest)

@app.route('/download')
def download():
    if not prediction_logs:
        return "No data to download", 404
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=prediction_logs[0].keys())
    writer.writeheader()
    writer.writerows(prediction_logs)
    return send_file(io.BytesIO(output.getvalue().encode()), mimetype='text/csv',
                     download_name='prediction_logs.csv', as_attachment=True)

@app.route('/model-info')
def model_info():
    from sklearn.metrics import mean_squared_error, r2_score
    X = data[["Temperature", "Humidity", "Solar", "Appliances", "Income"]]
    y = data["Energy_kWh"] if "Energy_kWh" in data.columns else data.iloc[:, 0]
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    return render_template("model_info.html", r2=round(r2, 3), mse=round(mse, 3))

@app.route('/charts')
def charts():
    timestamps = [r['timestamp'] for r in prediction_logs]
    predictions = [float(r['predicted_kWh']) for r in prediction_logs]
    return render_template("chart_dashboard.html", timestamps=timestamps, predictions=predictions)

@app.route('/compare')
def compare():
    timestamps = [r['timestamp'] for r in prediction_logs]
    predictions = [float(r['predicted_kWh']) for r in prediction_logs]
    actuals = [float(r.get('actual_kWh') or 0) for r in prediction_logs]
    return render_template("compare_chart.html", timestamps=timestamps, predictions=predictions, actuals=actuals)

@app.route('/agent-decision', methods=['POST'])
def agent_decision():
    input_data = request.get_json()
    result = run_energy_agent(input_data)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)

