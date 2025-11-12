import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


def create_fuzzy_variables():
    # Soil moisture (%)
    moisture = ctrl.Antecedent(np.arange(0, 101, 1), "moisture")
    moisture["dry"] = fuzz.trimf(moisture.universe, [15, 20, 30])
    moisture["ideal"] = fuzz.trimf(moisture.universe, [30, 45, 60])
    moisture["wet"] = fuzz.trimf(moisture.universe, [60, 70, 85])

    # Soil pH
    ph = ctrl.Antecedent(np.arange(4, 9, 0.01), "pH")
    ph["acidic"] = fuzz.trimf(ph.universe, [4.5, 5.0, 5.8])
    ph["ideal"] = fuzz.trimf(ph.universe, [6.0, 6.3, 6.8])
    ph["alkaline"] = fuzz.trimf(ph.universe, [6.8, 7.5, 8.5])

    # Nitrogen (mg/kg)
    nitrogen = ctrl.Antecedent(np.arange(0, 251, 1), "nitrogen")
    nitrogen["low"] = fuzz.trimf(nitrogen.universe, [0, 20, 40])
    nitrogen["adequate"] = fuzz.trimf(nitrogen.universe, [50, 80, 100])
    nitrogen["high"] = fuzz.trimf(nitrogen.universe, [150, 200, 250])

    # Temperature (°C)
    temperature = ctrl.Antecedent(np.arange(15, 42, 0.1), "temperature")
    temperature["cool"] = fuzz.trimf(temperature.universe, [15, 18, 22])
    temperature["ideal"] = fuzz.trimf(temperature.universe, [22, 24, 26])
    temperature["hot"] = fuzz.trimf(temperature.universe, [26, 30, 38])

    # Humidity (%)
    humidity = ctrl.Antecedent(np.arange(20, 101, 0.1), "humidity")
    humidity["dry"] = fuzz.trimf(humidity.universe, [40, 50, 60])
    humidity["ideal"] = fuzz.trimf(humidity.universe, [60, 65, 70])
    humidity["humid"] = fuzz.trimf(humidity.universe, [70, 80, 90])

    return moisture, ph, nitrogen, temperature, humidity


if __name__ == "__main__":
    create_fuzzy_variables()

