# models/demand_model.py
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

class DemandForecastModel:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model_path = 'models/saved_model/demand_model.pkl'

    def train(self, X, y):
        self.model.fit(X, y)
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)
        print(f"Model saved to {self.model_path}")

    def predict(self, X):
        if not hasattr(self.model, 'estimators_'):
            self.model = joblib.load(self.model_path)
        return self.model.predict(X)