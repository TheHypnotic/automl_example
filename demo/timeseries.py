
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


# --------------     to load model -----------------
# model_name = "Autoformer"
# model_id = manager.get_model_id_by_name(model_name)

model_name = "TimesNet"
model_id = manager.get_model_id_by_name(model_name)

manager.get_model(
    model_name= model_name,  # or any valid model ID
    local_dest="."
)
# ---------------------------------------------------


#----------------- passengers dataset ----------------

cfg = {
    "task": "timeseries",
    "epochs": 10,
    "batch_size": 16,
    "seq_len": 12,
    "pred_len": 1,
    "input_channels": 1,  # Added this
    "output_size": 1,     # Added this
    "dataset_config": {
        "source": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv",
    },

    # Model save
    # "save_model": True,
    # "model_dir": "model/",

    # "load_model": True,  
    # "model_dir": f"./{model_id}/",




    "model_config": {
        "type": "tslib",
        "name": "Autoformer",  # Try "InceptionTime", "FCNPlus", "ResNetPlus", etc.
        "task_name": "long_term_forecast",  # ✅ corrected task

    }
}

# cfg = {
#     "task": "timeseries",
#     "epochs": 10,
#     "batch_size": 16,
#     "seq_len": 12,
#     "pred_len": 1,
#     "input_channels": 1,
#     "output_size": 1,
#     "dataset_config": {
#         "source": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv",
#         "target_column": 'Passengers'
#     },

    # Model save
    # "save_model": True,
    # "model_dir": "model/",

    # "load_model": True,  
    # "model_dir": f"./{model_id}/",

#     "model_config": {
#         "type": "tslib",
#         # "name": "Autoformer",
#         # "name": "Reformer",
#         "name": "TimesNet",
#         "task_name": "long_term_forecast",  # ✅ corrected task
#     }
# }
# ---------------------------------------------------

# --------------------oil spill dataset--------------------


# cfg = {
#     "task": "timeseries",
#     "epochs": 10,
#     "batch_size": 16,
#     "seq_len": 12,  # A sequence length of 30 days is a good starting point
#     "pred_len": 7,  # Predicting one week ahead
#     "input_channels": 1,
#     "output_size": 1,
#     "dataset_config": {
#         "source": "https://raw.githubusercontent.com/jbrownlee/Datasets/refs/heads/master/oil-spill.csv",
#     },

    # Model save
    # "save_model": True,
    # "model_dir": "model/",

    # "load_model": True,  
    # "model_dir": f"./{model_id}/",
#     "model_config": {
#         "type": "tslib",
#         "name": "Autoformer",
#         "task_name": "long_term_forecast",
#     }
# }

cfg = {
    "task": "timeseries",
    "epochs": 10,
    "batch_size": 16,
    "seq_len": 12,  # A sequence length of 30 days is a good starting point
    "pred_len": 7,  # Predicting one week ahead
    "input_channels": 1,
    "output_size": 1,
    "dataset_config": {
        "source": "https://raw.githubusercontent.com/jbrownlee/Datasets/refs/heads/master/oil-spill.csv",
    },

    # Model save
    # "save_model": True,
    # "model_dir": "model/",

    "load_model": True,  
    "model_dir": f"./{model_id}/",

    "model_config": {
        "type": "tslib",
        "name": "TimesNet",
        "task_name": "long_term_forecast",
    }
}
trainer = AutoTrainer(config=cfg)

trainer.run()

# local_model_id = manager.add_model(
#     source_type="local",
#     source_path="model/",
#     model_name="TimesNet",
#     # model_name="Autoformer",
#     code_path="." , # ← Replace with the path to your model.py if you have it
# )
