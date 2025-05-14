import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import matplotlib.pyplot as plt
import abc
import os
from PIL import Image


from clearml import Task, Dataset, Logger, OutputModel

# ----------------- Model Definitions -----------------
class AbstractModelArchitecture(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def forward(self, x):
        pass

class LSTMRegressorArchitecture(AbstractModelArchitecture):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output, (hn, cn) = self.lstm(x)
        last_output = output[:, -1, :]
        return self.fc(last_output)

class AbstractTimeSeriesModel(nn.Module):
    def __init__(self, architecture: AbstractModelArchitecture):
        super().__init__()
        self.model = architecture

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, loss_fn):
        x, y = batch
        y_hat = self(x)
        return loss_fn(y_hat, y)

    def validation_step(self, batch, loss_fn):
        x, y = batch
        y_hat = self(x)
        return loss_fn(y_hat, y)

# ----------------- Trainer -----------------
class TimeSeriesTrainer:
    def __init__(self, model, train_loader, val_loader, optimizer, loss_fn, epochs=10, device=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.epochs = epochs
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.logger = Logger.current_logger()

    def prepare_batch(self, inputs, labels):
        return inputs.to(self.device), labels.to(self.device)

    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0.
        for i, batch in enumerate(self.train_loader):
            x, y = self.prepare_batch(*batch)
            y_hat = self.model(x)
            loss = self.loss_fn(y_hat, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
        avg_loss = total_loss / len(self.train_loader)
        print(f"Epoch {epoch+1} Training Loss: {avg_loss:.4f}")
        self.logger.report_scalar("Loss", "Train", value=avg_loss, iteration=epoch)
        return avg_loss

    def validate(self, epoch):
        self.model.eval()
        total_loss = 0.
        with torch.no_grad():
            for batch in self.val_loader:
                x, y = self.prepare_batch(*batch)
                y_hat = self.model(x)
                loss = self.loss_fn(y_hat, y)
                total_loss += loss.item()
        avg_loss = total_loss / len(self.val_loader)
        print(f"Epoch {epoch+1} Validation Loss: {avg_loss:.4f}")
        self.logger.report_scalar("Loss", "Validation", value=avg_loss, iteration=epoch)
        return avg_loss

    def run(self):
        for epoch in range(self.epochs):
            self.train_one_epoch(epoch)
            self.validate(epoch)

# ----------------- Data Loading -----------------
def load_clearml_dataset(params):
    dataset = Dataset.get(
        dataset_name=params["dataset_name"],
        dataset_project=params["dataset_project"]
    )
    local_path = dataset.get_local_copy()

    csv_files = [f for f in os.listdir(local_path) if f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError("No CSV file found in dataset directory")

    csv_path = os.path.join(local_path, csv_files[0])
    df = pd.read_csv(csv_path)

    values = df.iloc[:, -1].values.astype("float32").reshape(-1, 1)
    scaler = MinMaxScaler()
    values_scaled = scaler.fit_transform(values)

    X, Y = [], []
    for i in range(len(values_scaled) - params["sequence_length"] - params["prediction_length"]):
        X.append(values_scaled[i:i + params["sequence_length"]])
        Y.append(values_scaled[i + params["sequence_length"]:i + params["sequence_length"] + params["prediction_length"]])

    X = np.array(X)
    Y = np.array(Y).reshape(-1, 1)

    split_idx = int((1 - params["test_split"]) * len(X))
    X_train, Y_train = X[:split_idx], Y[:split_idx]
    X_val, Y_val = X[split_idx:], Y[split_idx:]

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(Y_val, dtype=torch.float32))

    return train_dataset, val_dataset, scaler, X_val, Y_val

# ----------------- Main Execution -----------------
def main():
    print("🚀 Starting ClearML Time Series Training Task")
    task = Task.init(project_name='TimeSeries SDK', task_name='Train LSTM TimeSeries Model', task_type=Task.TaskTypes.training)

    params = {
        "dataset_name": "monthly_sunspots_dataset",
        "dataset_project": "TimeSeries SDK",
        "sequence_length": 12,
        "prediction_length": 1,
        "test_split": 0.2,
        "input_size": 1,
        "hidden_size": 32,
        "num_layers": 1,
        "output_size": 1,
        "learning_rate": 0.001,
        "batch_size": 16,
        "epochs": 5
    }

    params = task.connect(params)
    train_dataset, val_dataset, scaler, X_val_raw, Y_val_raw = load_clearml_dataset(params)

    train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=params["batch_size"])

    architecture = LSTMRegressorArchitecture(
        input_size=params["input_size"],
        hidden_size=params["hidden_size"],
        num_layers=params["num_layers"],
        output_size=params["output_size"]
    )

    model = AbstractTimeSeriesModel(architecture)

    trainer = TimeSeriesTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=torch.optim.Adam(model.parameters(), lr=params["learning_rate"]),
        loss_fn=nn.MSELoss(),
        epochs=params["epochs"]
    )

    trainer.run()

    # Save and upload model
    checkpoint_path = "checkpoints/lstm_timeseries.pt"
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)

    output_model = OutputModel(task=task, framework='pytorch')
    output_model.update_weights(checkpoint_path)

    # Log a sample prediction plot
    print("📊 Logging prediction plot to ClearML")
    model.eval()
    sample_inputs = torch.tensor(X_val_raw[:20], dtype=torch.float32).to(trainer.device)
    with torch.no_grad():
        predictions = model(sample_inputs).cpu().numpy()
    actual = Y_val_raw[:20]

    plt.figure(figsize=(10, 5))
    plt.plot(predictions, label="Predicted")
    plt.plot(actual, label="Actual")
    plt.legend()
    plt.title("LSTM Prediction vs Actual")
    plt.grid(True)
    plot_path = "prediction_plot.png"
    plt.savefig(plot_path)
    # img = Image.open(plot_path)
    # Logger.current_logger().report_image("Prediction", "Example Forecast", iteration=0, image=img)

    # Logger.current_logger().report_image("Prediction", "Example Forecast", iteration=0, image=plot_path)

    print("✅ Task Completed")

if __name__ == "__main__":
    main()
