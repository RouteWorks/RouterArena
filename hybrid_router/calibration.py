import math
import json

class TemperatureScaler:
    def __init__(self, temperatures: dict[str, float], default_temperature=1.0):
        """
        Takes a temperatures: dict[str, float] and stores it. 
        Also provides a default_temperature parameter defaulting to 1.0.
        This is what gets used for any model not yet in the dict (i.e. before calibration has been run).
        """
        self.temperatures = temperatures
        self.default_temperature = default_temperature

    def apply(self, model_name: str, raw_logit: float) -> float:
        """
        Returns sigmoid(logit / T) where T is looked up from the dict, falling back to default_temperature 
        if the model isn't there yet. 

        Sigmoid is implemented as 1 / (1 + exp(-x)) using math.exp.
        """
        if model_name not in self.temperatures:
            temperature = self.default_temperature
        else:
            temperature = self.temperatures[model_name]

        return 1 / (1 + math.exp(-1 * raw_logit / max(temperature, 1e-8)))

    def save(self, path: str):
        """
        Writes self.temperatures to a JSON file.
        """
        with open(path, 'w', encoding="utf-8") as file:
            json.dump(self.temperatures, file, indent=4)
    
    @classmethod
    def load(cls, path: str):
        """
        A class method that reads the JSON file and returns a TemperatureScaler instance.
        """
        with open(path, 'r') as file:
            data = json.load(file)

        return cls(data)