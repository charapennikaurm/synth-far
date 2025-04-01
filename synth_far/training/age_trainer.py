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

class AGETrainingConfig(BaseTrainingConfig):
    model: ModelConfig
    train_dataloader: DataLoaderConfig
    val_dataloader: DataLoaderConfig
    clip_grad_policy: Literal["norm", "value", "none"] = "none"
    clip_grad_value: float = 1.0
    loss_age: ObjectConfig
    loss_gender: ObjectConfig
    loss_ethnicity: ObjectConfig
    lambda_age: float = 1.0
    lambda_gender: float = 1.0
    lambda_ethnicity: float = 1.0

class AGETrainer(BaseTrainer):
    config_class = AGETrainingConfig

    def __init__(self, config: AGETrainingConfig):
        super().__init__(config)
        self.model, self.optimizer, self.scheduler = self.config.model.build()
        self.train_dataloader = self.config.train_dataloader.build()
        self.val_dataloader = self.config.val_dataloader.build()
        self.loss_age = self.config.loss_age.build()
        self.loss_gender = self.config.loss_gender.build()
        self.loss_ethnicity = self.config.loss_ethnicity.build()
        self.lambda_age = self.config.lambda_age
        self.lambda_gender = self.config.lambda_gender
        self.lambda_ethnicity = self.config.lambda_ethnicity

        self.age_metric = torchmetrics.regression.MeanAbsoluteError()
        self.gender_metric = torchmetrics.classification.BinaryAccuracy(0.5)
        self.ethnicity_metric = torchmetrics.classification.MulticlassAccuracy(num_classes=5)

        (
            self.model, 
            self.optimizer,
            self.scheduler, 
            self.train_dataloader, 
            self.val_dataloader, 
            self.loss_age,
            self.loss_gender,
            self.loss_ethnicity,
            self.age_metric,
            self.gender_metric,
            self.ethnicity_metric
        ) = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.scheduler, 
            self.train_dataloader, 
            self.val_dataloader, 
            self.loss_age, 
            self.loss_gender, 
            self.loss_ethnicity,
            self.age_metric,
            self.gender_metric,
            self.ethnicity_metric
        )

        self.train_dataloader_iterator = EndlessIterator(self.train_dataloader)
        

    def _train_step(self) -> Dict[str, Any]:
        self.model.train()
        self.optimizer.zero_grad()
        batch = next(self.train_dataloader_iterator)
        images, labels = batch
        preds = self.model(images)
        age_loss = self.loss_age(preds["age"], labels["age"])
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


    def _validate(self) -> Dict[str, Any]:
        self.model.eval()

        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validation", leave=False):
                images, labels = batch
                preds = self.model(images)
                self.age_metric(preds["age"], labels["age"])
                self.gender_metric(preds["gender"], labels["gender"])
                self.ethnicity_metric(preds["ethnicity"], labels["ethnicity"].long().flatten())

        metrics = {
            "val/age_MAE": self.age_metric.compute().item(),
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

    def _checkpoint(self, checkpoint_dir: str):
        if not self.accelerator.is_main_process:
            return
        super()._checkpoint(checkpoint_dir)
        model_path = os.path.join(checkpoint_dir, "model.pth")
        state_dict = self.accelerator.unwrap_model(self.model).state_dict()
        torch.save(state_dict, model_path)
