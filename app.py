from flask import Flask, request, jsonify, render_template
from enersense import generate_synthetic_data, train_consumption_model, forecast_solar_generation, optimize_grid

app = Flask(__name__)

data = generate_synthetic_data()
model = train_consumption_model(data)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.json
    df = data.iloc[[0]].copy()
    df["Temperature"] = input_data.get("temperature", 25)
    df["Humidity"] = input_data.get("humidity", 50)
    df["Solar"] = input_data.get("solar", 600)
    df["Appliances"] = input_data.get("appliances", 5)
    df["Income"] = input_data.get("income", 500)

    prediction = model.predict(df[["Temperature", "Humidity", "Solar", "Appliances", "Income"]])[0]
    return jsonify({"predicted_consumption_kWh": round(prediction, 2)})

@app.route('/predict_ui', methods=['POST'])
def predict_ui():
    df = data.iloc[[0]].copy()
    df["Temperature"] = float(request.form['temperature'])
    df["Humidity"] = float(request.form['humidity'])
    df["Solar"] = float(request.form['solar'])
    df["Appliances"] = int(request.form['appliances'])
    df["Income"] = float(request.form['income'])

    prediction = model.predict(df[["Temperature", "Humidity", "Solar", "Appliances", "Income"]])[0]
    return f"<h3>Predicted Energy Consumption: {round(prediction, 2)} kWh</h3><a href='/'>Back</a>"

@app.route('/optimize', methods=['POST'])
def optimize():
    req = request.json
    demand = req.get("demand", 10)
    solar = req.get("solar", 8)
    result = optimize_grid(demand, solar)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)

@app.route('/')
def home():
    try:
        return render_template("index.html")
    except Exception as e:
        return f"<h2>Error loading home: {e}</h2>"

