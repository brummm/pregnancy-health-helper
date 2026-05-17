import joblib
import pandas as pd
import os

class HealthRiskPredictor:
    def __init__(self, model_path=None):
        if model_path is None:
            # Assume model is in the training folder for now, but in production we might move it
            model_path = os.path.join(os.path.dirname(__file__), '../../training/maternal_health_model.joblib')
        
        self.model = joblib.load(model_path)
        self.risk_mapping = {0: 'low risk', 1: 'mid risk', 2: 'high risk'}

    def predict(self, data):
        """
        data: dict containing Age, SystolicBP, DiastolicBP, BS, BodyTemp, HeartRate
        """
        # Ensure correct column order
        cols = ['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate']
        
        df = pd.DataFrame([data])[cols]
        prediction = self.model.predict(df)[0]
        return self.risk_mapping[prediction]
