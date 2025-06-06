import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import abc
import os

from clearml import Dataset
from clearml import Task, OutputModel
from clearml import Logger
from clearml.automation import UniformParameterRange, DiscreteParameterRange, HyperParameterOptimizer


# Initialize ClearML Task
task = Task.init(project_name='TimeSeries test',
                 task_name='timeseries',
                 task_type=Task.TaskTypes.training)

# ----- load dataset from ClearML -----
def load_clearml_dataset(params):
    """
    Load and preprocess a time series dataset from ClearML.
    
    Parameters:
        params (dict): Should include the following keys:
            - dataset_name: str
            - dataset_project: str
            - sequence_length: int
            - prediction_length: int
            - test_split: float (0 < test_split < 1)

    Returns:
        train_dataset, val_dataset, scaler
    """
    dataset = Dataset.get(
        dataset_name=params["dataset_name"],
        dataset_project=params["dataset_project"]
    )
    local_path = dataset.get_local_copy()
    
    # Automatically find the first CSV in the dataset directory
    csv_files = [f for f in os.listdir(local_path) if f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError("No CSV file found in dataset directory")
    
    csv_path = os.path.join(local_path, csv_files[0])
    df = pd.read_csv(csv_path)
    
    # Use the last column as the target time series
    values = df.iloc[:, -1].values.astype("float32").reshape(-1, 1)

    scaler = MinMaxScaler()
    values_scaled = scaler.fit_transform(values)

    seq_len = params["sequence_length"]
    pred_len = params["prediction_length"]
    test_split = params["test_split"]

    X, Y = [], []
    for i in range(len(values_scaled) - seq_len - pred_len):
        X.append(values_scaled[i:i + seq_len])
        Y.append(values_scaled[i + seq_len:i + seq_len + pred_len])

    X = np.array(X)
    Y = np.array(Y).reshape(-1, 1)  # flatten targets

    split_idx = int((1 - test_split) * len(X))
    X_train, Y_train = X[:split_idx], Y[:split_idx]
    X_val, Y_val = X[split_idx:], Y[split_idx:]

    X_train = torch.tensor(X_train, dtype=torch.float32)
    Y_train = torch.tensor(Y_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    Y_val = torch.tensor(Y_val, dtype=torch.float32)

    train_dataset = TensorDataset(X_train, Y_train)
    val_dataset = TensorDataset(X_val, Y_val)

    return train_dataset, val_dataset, scaler


# ----------------------------- Model Definitions -----------------------------
class AbstractModelArchitecture(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def forward(self, x):
        pass

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

class LSTMRegressorArchitecture(AbstractModelArchitecture):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output, (hn, cn) = self.lstm(x)
        last_output = output[:, -1, :]
        return self.fc(last_output)

class GRURegressorArchitecture(AbstractModelArchitecture):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output, hn = self.gru(x)
        last_output = output[:, -1, :]
        return self.fc(last_output)

# ----------------------------- Trainer -----------------------------
class AbstractTrainer(abc.ABC):
    def __init__(self, model, train_loader, val_loader, optimizer, loss_fn, epochs=10,
                 device=None, log_dir=None, checkpoint_path=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.epochs = epochs
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.clearml_logger = Logger.current_logger()

        self.checkpoint_path = checkpoint_path or f'model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pt'
        self.writer = SummaryWriter(log_dir or f'runs/train_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        self.best_val_loss = float('inf')

    def prepare_batch(self, inputs, labels):
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        return inputs, labels

    @abc.abstractmethod
    def training_step(self, batch): pass

    @abc.abstractmethod
    def validation_step(self, batch): pass

    def train_one_epoch(self, epoch_index):
        self.model.train()
        running_loss = 0.

        for i, batch in enumerate(self.train_loader):
            loss = self.training_step(batch)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()

        return running_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        running_vloss = 0.0
        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                loss = self.validation_step(batch)
                running_vloss += loss
        return running_vloss / (i + 1)

    def run(self):
        for epoch in range(self.epochs):
            print(f'EPOCH {epoch + 1}/{self.epochs}:')
            train_loss = self.train_one_epoch(epoch)
            val_loss = self.validate()
            print(f'LOSS train {train_loss:.4f} valid {val_loss:.4f}')

            self.writer.add_scalars("Loss", {"Train": train_loss, "Validation": val_loss}, epoch + 1)
            self.clearml_logger.report_scalar("Loss", "Train", value=train_loss, iteration=epoch)
            self.clearml_logger.report_scalar("Loss", "Validation", value=val_loss, iteration=epoch)
            
            self.model.eval()
            preds, targets = [], []
            with torch.no_grad():
                for x_batch, y_batch in self.val_loader:
                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    y_pred = self.model(x_batch)
                    preds.append(y_pred.cpu().numpy())
                    targets.append(y_batch.cpu().numpy())

            preds = np.concatenate(preds).flatten()
            targets = np.concatenate(targets).flatten()
            preds_rescaled = self.scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
            targets_rescaled = self.scaler.inverse_transform(targets.reshape(-1, 1)).flatten()

            mse = mean_squared_error(targets_rescaled, preds_rescaled)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(targets_rescaled, preds_rescaled)
            r2 = r2_score(targets_rescaled, preds_rescaled)

            self.writer.add_scalar("Metrics/RMSE", rmse, epoch + 1)
            self.writer.add_scalar("Metrics/MAE", mae, epoch + 1)
            self.writer.add_scalar("Metrics/R2", r2, epoch + 1)

            self.clearml_logger.report_scalar("Metrics", "RMSE", value=rmse, iteration=epoch)
            self.clearml_logger.report_scalar("Metrics", "MAE", value=mae, iteration=epoch)
            self.clearml_logger.report_scalar("Metrics", "R2", value=r2, iteration=epoch)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
                torch.save(self.model.state_dict(), self.checkpoint_path)

        self.writer.close()

class TimeSeriesTrainer(AbstractTrainer):
    def training_step(self, batch):
        x, y = self.prepare_batch(*batch)
        y_hat = self.model(x)
        return self.loss_fn(y_hat, y)

    def validation_step(self, batch):
        x, y = self.prepare_batch(*batch)
        y_hat = self.model(x)
        return self.loss_fn(y_hat, y)

def build_model(params):
    model_type = params["model_type"].lower()

    if model_type == "lstm":
        architecture = LSTMRegressorArchitecture(
            input_size=params["input_size"],
            hidden_size=params["hidden_size"],
            num_layers=params["num_layers"],
            output_size=params["output_size"]
        )
    elif model_type == "gru":
        architecture = GRURegressorArchitecture(
            input_size=params["input_size"],
            hidden_size=params["hidden_size"],
            num_layers=params["num_layers"],
            output_size=params["output_size"]
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return AbstractTimeSeriesModel(architecture)

# ----------------------------- Execution -----------------------------
logger = Logger.current_logger()

params = {
    # "dataset_name": "airline_passengers_dataset",
    "dataset_name": "oil_price",
    "dataset_project": "demo",
    "sequence_length": 12,
    "prediction_length": 1,
    "test_split": 0.2,
    "input_size": 1,
    "hidden_size": 32,
    "num_layers": 1,
    "output_size": 1,
    "learning_rate": 0.001,
    "batch_size": 16,
    "epochs": 10,
    "model_type": "gru"
}

# Connect the parameters to ClearML (allows UI editing)
params = task.connect(params)

# Load dataset from ClearML
train_dataset, val_dataset, scaler = load_clearml_dataset(params)
train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=params["batch_size"])

# Define the model architecture
# architecture = LSTMRegressorArchitecture(
#     input_size=params["input_size"],
#     hidden_size=params["hidden_size"],
#     num_layers=params["num_layers"],
#     output_size=params["output_size"]
# )

# model = AbstractTimeSeriesModel(architecture)
model = build_model(params)

trainer = TimeSeriesTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=torch.optim.Adam(model.parameters(), lr=params["learning_rate"]),
    loss_fn=nn.MSELoss(),
    epochs=params["epochs"],
    log_dir="runs/timeseries",
    checkpoint_path="checkpoints/lstm_timeseries.pt"
)
trainer.scaler = scaler


# ------ log and plot
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from PIL import Image


def evaluate_and_log_results(model, val_loader, scaler, logger, writer, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            y_pred = model(x_batch)
            preds.append(y_pred.cpu().numpy())
            targets.append(y_batch.cpu().numpy())

    preds = np.concatenate(preds).flatten()
    targets = np.concatenate(targets).flatten()

    # Reverse MinMax scaling
    preds_rescaled = scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
    targets_rescaled = scaler.inverse_transform(targets.reshape(-1, 1)).flatten()

    # Metrics
    mse = mean_squared_error(targets_rescaled, preds_rescaled)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(targets_rescaled, preds_rescaled)
    r2 = r2_score(targets_rescaled, preds_rescaled)

    print(f"\nEvaluation Metrics:\nRMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

    # Log metrics
    # logger.report_scalar("Metrics", "RMSE", value=rmse, iteration=0)
    # logger.report_scalar("Metrics", "MAE", value=mae, iteration=0)
    # logger.report_scalar("Metrics", "R2", value=r2, iteration=0)
    # writer.add_scalar("Metrics/RMSE", rmse)
    # writer.add_scalar("Metrics/MAE", mae)
    # writer.add_scalar("Metrics/R2", r2)

    # Plot predictions vs actuals
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(targets_rescaled, label="Actual")
    ax.plot(preds_rescaled, label="Predicted")
    ax.set_title("Predicted vs Actual")
    ax.legend()
    fig_path = "plots/pred_vs_actual.png"
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path)
    # logger.report_image("Predictions", "Pred vs Actual", iteration=0, image=fig_path) 
    logger.report_image("Predictions", "Pred vs Actual", iteration=0, image=Image.open(fig_path))

    plt.close(fig)

    # Plot residuals
    residuals = targets_rescaled - preds_rescaled
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(residuals, bins=50)
    ax.set_title("Residuals Histogram")
    fig_path = "plots/residuals.png"
    plt.savefig(fig_path)
    # logger.report_image("Predictions", "Residuals", iteration=0, image=fig_path)
    logger.report_image("Predictions", "Residuals", iteration=0, image=Image.open(fig_path))

    plt.close(fig)


if __name__ == "__main__":
    trainer.run()

    # Log the trained model to ClearML
    output_model = OutputModel(task=task, framework='pytorch')
    output_model.update_weights(trainer.checkpoint_path)

    # Evaluate and log predictions, residuals, metrics
    evaluate_and_log_results(trainer.model, val_loader, scaler, logger, trainer.writer, trainer.device)
