import torch
import torch.nn as nn
from torch.nn import functional as F
from ml_trainer import AutoTrainer
from ml_trainer.base import AbstractModelArchitecture

# Simple CNN model
class SimpleCNN(nn.Module):
    
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

cfg = {
    "task": "image_classification",
    "batch_size": 64,
    "split_ratio": 0.8,
    "lr": 1e-3,
    "epochs": 5,
    "num_classes": 10,  # override if needed after loading


    "dataset_config": {
        "source": "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",  # or "./data/my_dataset"
        "transform_config": {
            "resize": [32, 32],
            "normalize_mean": [0.5, 0.5, 0.5],
            "normalize_std": [0.5, 0.5, 0.5]
        }
    },
    "model_config": {
        "type": "timm",
        "name": "resnet50",
        "pretrained": True
    }
}

trainer = AutoTrainer(config=cfg)
trainer.run()
