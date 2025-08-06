# enersense.py
# ML and Optimization logic for EnerSense AI Agent

try:
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    from pulp import LpProblem, LpMinimize, LpVariable
except ImportError as e:
    print("Missing module:", e)
    print("Please install required packages with:")
    print("pip install numpy pandas scikit-learn pulp")
    exit()

# ------------------ Synthetic Data Generator ------------------
def generate_synthetic_data(n=1000):
    np.random.seed(42)
    temperature = np.random.uniform(15, 40, n)
    humidity = np.random.uniform(20, 90, n)
    wind_speed = np.random.uniform(0, 20, n)
    solar_irradiance = np.random.uniform(200, 1000, n)
    income = np.random.uniform(100, 1000, n)
    appliances = np.random.randint(1, 10, n)

    energy_consumption = (
        0.5 * temperature - 0.2 * humidity + 0.3 * solar_irradiance/100 +
        0.1 * appliances - 0.0002 * income + np.random.normal(0, 2, n)
    )

    data = pd.DataFrame({
        "Temperature": temperature,
        "Humidity": humidity,
        "Wind": wind_speed,
        "Solar": solar_irradiance,
        "Income": income,
        "Appliances": appliances,
        "Consumption": energy_consumption
    })
    return data

# ------------------ Train Consumption Prediction Model ------------------
def train_consumption_model(data):
    X = data[["Temperature", "Humidity", "Solar", "Appliances", "Income"]]
    y = data["Consumption"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("R² Score:", r2_score(y_test, y_pred))
    print("MSE:", mean_squared_error(y_test, y_pred))

    return model

# ------------------ Forecast Solar Generation ------------------
def forecast_solar_generation(solar_irradiance):
    efficiency = 0.15
    panel_area = 10  # m^2
    energy_output = solar_irradiance * efficiency * panel_area / 1000  # in kWh
    return energy_output

# ------------------ Grid Optimization ------------------
def optimize_grid(demand, generation, cost_grid=0.15, cost_renewable=0.05):
    prob = LpProblem("Energy_Distribution", LpMinimize)

    x_grid = LpVariable("Grid_Energy", lowBound=0)
    x_renewable = LpVariable("Renewable_Energy", lowBound=0)

    prob += cost_grid * x_grid + cost_renewable * x_renewable
    prob += x_grid + x_renewable >= demand
    prob += x_renewable <= generation

    prob.solve()

    return {
        "Grid Energy (kWh)": x_grid.varValue,
        "Renewable Energy (kWh)": x_renewable.varValue,
        "Total Cost ($)": round(cost_grid * x_grid.varValue + cost_renewable * x_renewable.varValue, 2)
    }