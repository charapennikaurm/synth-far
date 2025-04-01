from .trainer import BaseTrainer
from .config import BaseTrainingConfig, ModelConfig, DataLoaderConfig, ObjectConfig
from typing import Literal, Dict, Any
import torch
from ..utils.torch_utils import EndlessIterator
import os
import torchmetrics
import torchmetrics.classification
import torchmetrics.regression
from tqdm.autonotebook import tqdm

from .age_trainer import AGETrainingConfig, AGETrainer  

class FFTrainingConfig(AGETrainingConfig):
    pass

class FFTrainer(AGETrainer):
    config_class = FFTrainingConfig

    def __init__(self, config: FFTrainingConfig):
        super().__init__(config)
        self.age_metric = torchmetrics.classification.MulticlassAccuracy(num_classes=9)
        self.age_metric = self.accelerator.prepare(self.age_metric)
        
    def _validate(self) -> Dict[str, Any]:
        self.model.eval()

        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validation", leave=False):
                images, labels = batch
                preds = self.model(images)
                self.age_metric(preds["age"], labels["age"].long().flatten())
                self.gender_metric(preds["gender"], labels["gender"])
                self.ethnicity_metric(preds["ethnicity"], labels["ethnicity"].long().flatten())

        metrics = {
            "val/age_acc": self.age_metric.compute().item(),
            "val/gender_acc": self.gender_metric.compute().item(),
            "val/ethnicity_acc": self.ethnicity_metric.compute().item(),
        }
        self.accelerator.log(
            metrics,
            step=self.global_step,
        )
        self.age_metric.reset()
        self.gender_metric.reset()
        self.ethnicity_metric.reset()
        return metrics


    def _train_step(self) -> Dict[str, Any]:
        self.model.train()
        self.optimizer.zero_grad()
        batch = next(self.train_dataloader_iterator)
        images, labels = batch
        preds = self.model(images)
        age_loss = self.loss_age(preds["age"], labels["age"].flatten())
        gender_loss = self.loss_gender(preds["gender"], labels["gender"])
        ethnicity_loss = self.loss_ethnicity(preds["ethnicity"], labels["ethnicity"].flatten())
        loss = self.lambda_age * age_loss + self.lambda_gender * gender_loss + self.lambda_ethnicity * ethnicity_loss
        self.accelerator.backward(loss)
        if self.accelerator.sync_gradients:
            if self.config.clip_grad_policy == "value":
                self.accelerator.clip_grad_value_(
                    self.model.parameters(), self.config.clip_grad_value
                )
            elif self.config.clip_grad_policy == "norm":
                self.accelerator.clip_grad_norm_(
                    self.model.parameters(), self.config.clip_grad_value
                )
        self.optimizer.step()
        self.scheduler.step()
        self.accelerator.log(
            {
                "train/loss": loss.item(),
                "train/age_loss": age_loss.item(),
                "train/gender_loss": gender_loss.item(),
                "train/ethnicity_loss": ethnicity_loss.item(),
                "optimizer/lr": self.optimizer.param_groups[0]["lr"],
            },
            step=self.global_step,
        )
        return {
            "train/loss": loss.item(),
        }
