def run_energy_agent(data):
    """
    Makes a smart decision about energy usage.
    Input: dict with 'temperature', 'humidity', 'solar', 'appliances', 'income'
    """
    solar = data.get("solar", 500)
    appliances = data.get("appliances", 5)
    income = data.get("income", 500)

    if solar > 600 and income > 400:
        return {
            "turn_on": True,
            "advice": "Use solar power now. It's optimal!",
            "solar_level": solar
        }
    elif solar < 300:
        return {
            "turn_on": False,
            "advice": "Save energy – solar generation is low.",
            "solar_level": solar
        }
    else:
        return {
            "turn_on": appliances <= 2,
            "advice": "Limit appliance use.",
            "solar_level": solar
        }
