
import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F
from dotenv import load_dotenv

from ml_trainer import AutoTrainer
from ml_trainer.base import AbstractModelArchitecture
from aipmodel.model_registry import MLOpsManager

load_dotenv()

# --------- fetch model from model registry --------
manager = MLOpsManager(
    clearml_url="http://web.mlops.ai-lab.ir/api_old",
    clearml_access_key="9E280YOU7E94HR84OMDPW71JBP2XPZ",
    clearml_secret_key="uRzPKO078_7TGFJRzImk0Zj1AwefwHgpiV11IWr8joS5M6AzKhwTTKI_HEHhxylXPoA",
    clearml_username="mlops-admin"
)

# ---------- Variables -------------
dataset='cifar-10'
epochs = 1
batch_size = 64
split_ratio = 0.8
lr = 0.01
save_model = False
load_model = True
model_name = "resnet50"
model_id = "resnet50_t"
model_save_name="resenet50_save1"
transform = {
            "resize": [32, 32],
            "normalize_mean": [0.5, 0.5, 0.5],
            "normalize_std": [0.5, 0.5, 0.5]
        }

# ---------
# Ensure valid model name/id
if model_name not in ["resnet50", "efficientnet_b0"] and model_id not in ["resnet50_t", "efficientnet_b0"]:
    raise ValueError("Invalid model name/id: choose from resnet50 or efficientnet_b0")

# Dataset configuration
if dataset == 'cifar-10':
    sources = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
elif dataset == "stl10":
    sources = "http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz"
else:
    raise ValueError("Invalid dataset: choose either 'cifar-10' or 'stl10'")

# --------------     to load model -----------------

if load_model: 
    model_id = manager.get_model_id_by_name(model_id)

    manager.get_model(
        model_name= model_name,  # or any valid model ID
        local_dest="."
)

#----------------- main config ----------------
cfg = {
    # Training Params
    "task": "image_classification",
    "batch_size": batch_size,
    "split_ratio": split_ratio,
    "lr": lr,
    "epochs": epochs,
    "num_classes": 10,  

    # Dataset
    "dataset_config": {
        "source": sources,
        "transform_config": transform
    },
    
    # Model save
    "save_model": save_model,
    "model_dir": "model/",

    "load_model": load_model,  
    "model_dir": f"./{model_id}/",
    

    # Model load
    "model_config": {
        "type": "timm",
        "name": model_name,
        "pretrained": True
    }
}



trainer = AutoTrainer(config=cfg)

trainer.run()

if save_model:
    local_model_id = manager.add_model(
        source_type="local",
        source_path="model/",
        model_name=model_save_name,
        code_path="." , # ← Replace with the path to your model.py if you have it
    )
