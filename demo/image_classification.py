
import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F
from dotenv import load_dotenv

from ml_trainer import AutoTrainer
from ml_trainer.base import AbstractModelArchitecture
from aipmodel.model_registry import MLOpsManager

load_dotenv()

manager = MLOpsManager(
    clearml_url="http://web.mlops.ai-lab.ir/api_old",
    clearml_access_key="9E280YOU7E94HR84OMDPW71JBP2XPZ",
    clearml_secret_key="uRzPKO078_7TGFJRzImk0Zj1AwefwHgpiV11IWr8joS5M6AzKhwTTKI_HEHhxylXPoA",
    clearml_username="mlops-admin"
)


# model_name = "resnet50_t"
# model_id = manager.get_model_id_by_name(model_name)

model_name = "efficientnet_b0"
model_id = manager.get_model_id_by_name(model_name)

manager.get_model(
    model_name= model_name,  # or any valid model ID
    local_dest="."
)

cfg = {
    # Training Params
    "task": "image_classification",
    "batch_size": 64,
    "split_ratio": 0.8,
    "lr": 1e-3,
    "epochs": 1,
    "num_classes": 10,  # override if needed after loading

    # Dataset
    "dataset_config": {
        # "source": "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",  # or "./data/my_dataset"
        "source": "http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz",
        "transform_config": {
            "resize": [32, 32],
            "normalize_mean": [0.5, 0.5, 0.5],
            "normalize_std": [0.5, 0.5, 0.5]
        }
    },
    
    # Model save
    # "save_model": True,
    # "model_dir": "model/",

    "load_model": True,  
    "model_dir": f"./{model_id}/",
    

    # Model load
    "model_config": {
        "type": "timm",
        # "name": "resnet50",
        "name": "efficientnet_b0",
        "pretrained": True
    }
}



trainer = AutoTrainer(config=cfg)

trainer.run()

# local_model_id = manager.add_model(
#     source_type="local",
#     source_path="model/",
#     # model_name="resnet50_t",
#     model_name="efficientnet_b0",
#     code_path="." , # ← Replace with the path to your model.py if you have it
# )
