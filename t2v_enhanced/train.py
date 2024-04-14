# General
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
from t2v_enhanced.model.video_ldm import VideoLDM
from model.callbacks import SaveConfigCallback
from inference_utils import CustomCLI
from pytorch_lightning import LightningDataModule

# Utilities
from inference_utils import *
from model_init import *
from model_func import *

if __name__ == "__main__":
    cli = CustomCLI(VideoLDM, LightningDataModule, subclass_mode_data=True,
                        auto_configure_optimizers=False, parser_kwargs={"parser_mode": "omegaconf"}, save_config_callback=SaveConfigCallback, save_config_kwargs={"log_dir": "training_results", "overwrite": True})
