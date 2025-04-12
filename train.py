import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
import joblib

class ORCO_Net(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, output_dim=20):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.log_softmax(self.fc2(x), dim=1)

def train_model(data_file, epochs=50, lr=0.001):
    df = pd.read_csv(data_file)
    X = df[['CA', 'CB']].values.astype('float32')
    y = df['label'].values.astype('int64')

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    model = ORCO_Net()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.NLLLoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = criterion(output, y_tensor)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "orco_model.pt")
    joblib.dump(scaler, "orco_scaler.pkl")
    print("✅ Model and scaler saved.")

if __name__ == "__main__":
    train_model("bmrb_4769_labeled.csv")