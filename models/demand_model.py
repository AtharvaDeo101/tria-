import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

class DemandForecastModel:
    def __init__(self, look_back=5, input_size=4, hidden_size=50):
        self.look_back = look_back
        self.input_size = input_size  # Number of features
        self.hidden_size = hidden_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        """Build the LSTM model using PyTorch."""
        class LSTMModel(nn.Module):
            def __init__(self, input_size, hidden_size):
                super(LSTMModel, self).__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True)
                self.fc = nn.Linear(hidden_size, 1)

            def forward(self, x):
                out, _ = self.lstm(x)
                out = self.fc(out[:, -1, :])  # Take the last time step
                return out

        return LSTMModel(input_size=self.input_size, hidden_size=self.hidden_size)

    def train(self, X, y, epochs=5, batch_size=32):
        """Train the model."""
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        optimizer = Adam(self.model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model.train()
        for epoch in range(epochs):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    def predict(self, X):
        """Make predictions."""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            predictions = self.model(X_tensor).cpu().numpy()
        return predictions

    def save(self, path):
        """Save the trained model."""
        torch.save(self.model.state_dict(), path)

    @staticmethod
    def load(path, look_back=5, input_size=4, hidden_size=50):
        """Load a trained model."""
        instance = DemandForecastModel(look_back=look_back, input_size=input_size, hidden_size=hidden_size)
        instance.model.load_state_dict(torch.load(path))
        instance.model.eval()
        return instance