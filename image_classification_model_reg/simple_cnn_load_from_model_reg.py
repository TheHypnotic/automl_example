import torch
import torch.nn as nn
from torch.nn import functional as F
from dotenv import load_dotenv

from ml_trainer import AutoTrainer
from ml_trainer.base import AbstractModelArchitecture
from aipmodel.model_registry import MLOpsManager

load_dotenv()

# Simple CNN model
class SimpleCNN(nn.Module, AbstractModelArchitecture):
    
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()

        # TODO: INIT LOGIC
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):

        # TODO: Forward logic
        x = self.pool(F.relu(self.conv1(x)))   # [batch, 32, 32, 32]
        x = self.pool(F.relu(self.conv2(x)))   # [batch, 64, 16, 16]
        # print(x.shape)
        x = x.view(x.size(0), -1)              # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def save(self, path):
        """Save model weights to a file."""
        torch.save(self.state_dict(), path)

    def load(self, path, map_location=None):
        """Load model weights from a file."""
        state_dict = torch.load(path, map_location=map_location)
        self.load_state_dict(state_dict)

manager = MLOpsManager(
    clearml_url="http://213.233.184.112:30008",
    clearml_access_key="J8Y6B7M4HU7O51JTOVOMLSBN535MB0",
    clearml_secret_key="7SeCHSOGV-Lx1HpSKBe3vY3nt7L6tp9voDX9sNiRwpFpfw28-yfabeU9NkWOCPjMO-s",
    clearml_username="dario"
)


cfg = {
    # Training Params
    "task": "image_classification",
    "batch_size": 64,
    "split_ratio": 0.8,
    "lr": 1e-3,
    "epochs": 5,
    "num_classes": 10,  # override if needed after loading

    # Dataset
    "dataset_config": {
        "source": "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",  # or "./data/my_dataset"
        "transform_config": {
            "resize": [32, 32],
            "normalize_mean": [0.5, 0.5, 0.5],
            "normalize_std": [0.5, 0.5, 0.5]
        }
    },
    
    # Model 
    "load_model": True,  
    "model_dir": "./df1769ae6cd246dea97ee669fc24a13c/",

    "model_config": {
        "type": "timm",
        "name": "resnet50",
        "pretrained": True
    }
}

manager.get_model(
    model_name="CNNModel",  # or any valid model ID
    local_dest="."
)

model = SimpleCNN()
trainer = AutoTrainer(config=cfg, model=model)
trainer.run()