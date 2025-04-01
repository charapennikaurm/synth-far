import accelerate
from .config import BaseTrainingConfig
from ..utils import torch_utils
from ..utils.files import json_dump
from ..utils.logging import get_logger
import os
from abc import abstractmethod
from tqdm import tqdm
from typing import Dict, Any
logger = get_logger()

class BaseTrainer:
    config_class = BaseTrainingConfig

    @abstractmethod
    def _checkpoint(self, checkpoint_dir):
        "Checkpointing logic goes here"
        if not self.accelerator.is_main_process:
            return
        self.accelerator.save_state(os.path.join(checkpoint_dir, "accelerator_state"))
        json_dump(self.config.as_dict(), os.path.join(checkpoint_dir, "config.json"), indent=2)

    @abstractmethod
    def _validate(self) -> Dict[str, Any]:
        "Validation logic goes here"
        pass
    
    @abstractmethod
    def _train_step(self) -> Dict[str, Any]:
        "Train step logic goes here"
        pass

    def __init__(self, config: BaseTrainingConfig):
        self.config = config
        self.accelerator = None
        self.global_step = 0
        self.setup()

    @classmethod
    def from_config_file(cls, config_path):
        return cls(cls.config_class.from_file(config_path))
    
    def setup(self):
        torch_utils.set_determenistic(self.config.seed)
        self._setup_accelerator()
        self._make_directories()
        self._init_trackers()

    def _setup_accelerator(self):
        accelerator_dict = self.config.accelerator.as_dict()
        accelerator_dict["project_dir"] = self.config.logging_directory
        accelerator_dict.pop("dict_repr")
        accelerator = accelerate.Accelerator(
            **accelerator_dict
        )
        self.accelerator = accelerator
    
    def _init_trackers(self):
        self.accelerator.init_trackers(
            self.config.experiment_name,
        )
        json_dump(self.config.as_dict(), os.path.join(self.config.logging_directory, "config.json"), indent=2)

    def _make_directories(self):
        if self.accelerator.is_local_main_process:
            os.makedirs(self.config.experiment_directory, exist_ok=True)
            os.makedirs(self.config.logging_directory, exist_ok=True)
            if not self.config.skip_checkpoints:
                os.makedirs(self.config.checkpoints_directory, exist_ok=True)

    def run(self):
        pb = tqdm(desc="Step", total=self.config.total_steps)
        last_train_log = {}
        last_val_log = {}
        for _ in range(self.config.total_steps):
            last_train_log = self._train_step()
            pb.set_postfix({
                **last_train_log, **last_val_log
            })

            if self.is_checkpoint_needed:
                self.make_checkpoint()
            if self.is_validation_needed:
                last_val_log = self.run_validation() 
                pb.set_postfix({
                    **last_train_log, **last_val_log
                })
            self.global_step += 1
            pb.update(1)
        self.accelerator.end_training()

    def make_checkpoint(self):
        if self.config.skip_checkpoints:
            return
        if self.accelerator.is_local_main_process:
            logger.info(f"Making checkpoint at step: {self.global_step}")
            checkpoint_dir = self.get_checkpoint_directory()
            os.makedirs(checkpoint_dir, exist_ok=True)
            self._checkpoint(checkpoint_dir)
            return checkpoint_dir

    def get_checkpoint_directory(self):
        return os.path.join(self.config.checkpoints_directory, str(self.global_step))
    
    def run_validation(self):
        if self.config.skip_validation:
            return 
        if self.accelerator.is_local_main_process:
            logger.info(f"Starting Validation on step {self.global_step}")
            res = self._validate()
            if self.config.validate_on_checkpoint:
                self.make_checkpoint()
            return res
        
    @property
    def is_checkpoint_needed(self):
        return (
            (not self.config.skip_checkpoints) and 
            (((self.global_step + 1) % self.config.checkpoint_every_n_steps) == 0)
        )
    
    @property
    def is_validation_needed(self):
        return (
            (not self.config.skip_validation) and 
            ((self.global_step + 1) % self.config.validate_every_n_steps == 0)
        )