
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
dataset='airline-passengers'
epochs = 10
batch_size = 16
save_model = False
load_model = False
model_name = "Autoformer"
model_id = "Autoformer"

# Ensure valid model name/id
if model_name not in ["Autoformer", "TimesNet"] and model_id not in ["Autoformer", "TimesNet"]:
    raise ValueError("Invalid model name/id: choose from Autoformer or TimesNet")

# Dataset configuration
if dataset == 'airline-passengers':
    sources = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
elif dataset == "oil-spill":
    sources = "https://raw.githubusercontent.com/jbrownlee/Datasets/refs/heads/master/oil-spill.csv"
else:
    raise ValueError("Invalid dataset: choose either 'airline-passengers' or 'oil-spill'")

# --------------     to load model -----------------
if load_model: 
    model_id = manager.get_model_id_by_name(model_name)

    manager.get_model(
        model_name= model_name,
        local_dest="."
    )


#----------------- main config ----------------

cfg = {
    "task": "timeseries",
    "epochs": epochs,
    "batch_size": batch_size,
    "seq_len": 12,
    "pred_len": 1,
    "input_channels": 1,  # Added this
    "output_size": 1,     # Added this
    "dataset_config": {
        "source": sources,
    },

    # Model save
    "save_model": save_model,
    "model_dir": "model/",

    "load_model": load_model,  
    "model_dir": f"./{model_id}/",




    "model_config": {
        "type": "tslib",
        "name": model_name,  # Try "InceptionTime", "FCNPlus", "ResNetPlus", etc.
        "task_name": "long_term_forecast",  # ✅ corrected task

    }
}

trainer = AutoTrainer(config=cfg)

trainer.run()

if save_model:
    local_model_id = manager.add_model(
        source_type="local",
        source_path="model/",
        model_name="TimesNet",
        # model_name="Autoformer",
        code_path="." , # ← Replace with the path to your model.py if you have it
    )
