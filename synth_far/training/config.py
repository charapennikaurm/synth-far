from .. import utils
from pydantic import BaseModel
from typing import Dict
from typing import Optional, List, Optional, Union, Any
import os
from copy import deepcopy
import torch
from .scheduling import get_scheduler


logger = utils.logging.get_logger()

class BaseConfig(BaseModel):
    @classmethod
    def from_file(
        cls,
        path: str,
    ) -> "BaseConfig":
        extension = utils.files.get_extension(path)
        if extension in utils.files.YAML_EXTENSIONS:
            config_dict = utils.files.read_yaml(path)
        elif extension in utils.files.JSON_EXTENSIONS:
            config_dict = utils.files.read_json(path)
        else:
            raise ValueError(
                f"Unsupported extension {extension} for config file. "
                f"Use {utils.files.YAML_EXTENSIONS} for config file in yaml format "
                f"or {utils.files.JSON_EXTENSIONS} for json format."
            )

        return cls(**config_dict)

    def save(self, path):
        extension = utils.files.get_extension(path)
        if extension in utils.files.YAML_EXTENSIONS:
            utils.files.yaml_dump(self.as_dict(), path)
        elif extension in utils.files.JSON_EXTENSIONS:
            utils.files.json_dump(self.as_dict(), path, indent=2)
        else:
            raise ValueError(
                f"Unsupported extension {extension} for config file. "
                f"Use {utils.files.YAML_EXTENSIONS} for config file in yaml format "
                f"or {utils.files.JSON_EXTENSIONS} for json format."
            )

    def as_dict(self) -> Dict:
        return self.model_dump()

    def __str__(self) -> str:
        return str(self.as_dict())

    def __repr__(self) -> str:
        return str(self)

class ObjectConfig(BaseConfig, extra='allow'):
    type: str

    def build(self):
        return utils.torch_utils.dict_to_object(self.as_dict())
    
class AcceleratorConfig(BaseConfig):
    device_placement: bool = True
    gradient_accumulation_steps: int = 1
    cpu: bool = False
    mixed_precision: Optional[str] = None
    dataloader_config: Optional[ObjectConfig] = None
    deepspeed_plugin: Optional[ObjectConfig] = None
    fsdp_plugin: Optional[ObjectConfig] = None
    megatron_lm_plugin: Optional[ObjectConfig] = None
    rng_types: Optional[List[str]] = None
    log_with: Optional[Union[str, List[str]]] = None
    gradient_accumulation_plugin: Optional[ObjectConfig] = None
    step_scheduler_with_optimizer: bool = True
    kwargs_handlers: Optional[List[ObjectConfig]] = None
    dynamo_backend: Optional[str] = None
    dict_repr: Dict[str, Any] = {}

    def model_post_init(self, __context):
        self.dict_repr = {}
        for attr in [
            "dataloader_config", "deepspeed_plugin",
            "fsdp_plugin", "megatron_lm_plugin",
            "gradient_accumulation_plugin",
        ]:
            object_config = getattr(self, attr)
            self.dict_repr[attr] = object_config
            object_config = object_config.build() if object_config is not None else None
            setattr(self, attr, object_config)
        
        self.dict_repr['kwargs_handlers'] = self.kwargs_handlers
        self.kwargs_handlers = None if self.kwargs_handlers is None else (
            [handler.build() for handler in self.kwargs_handlers]
        )
        return super().model_post_init(__context)


    def as_dict(self):
        dict_ = super().as_dict()
        for k, v in self.dict_repr.items():
            dict_[k] = v
        return dict_

class SchedulerConfig(BaseConfig):
    name: str
    num_warmup_steps: Optional[int] = None
    num_training_steps: Optional[int] = None
    num_cycles: int = 1
    power: float = 1.0

class ModelConfig(BaseConfig):
    model: ObjectConfig
    optimizer: ObjectConfig
    scheduler: SchedulerConfig

    def build(self):
        """returns model, it's optimizer and lr scheduler"""
        model = self.build_model()
        optimizer = self.build_optimizer(model)
        scheduler = self.build_scheduler(optimizer)

        return model, optimizer, scheduler

    def build_model(self):
        d = deepcopy(self.model.as_dict())
        weights_path = d.pop("weights", None)
        model = utils.torch_utils.dict_to_object(d)
        if weights_path is not None:
            sd = torch.load(weights_path, "cpu")
            model.load_state_dict(sd)
        return model

    def build_optimizer(self, model):
        d = deepcopy(self.optimizer.as_dict())
        d["params"] = model.parameters()
        optimizer = utils.torch_utils.dict_to_object(d)
        return optimizer

    def build_scheduler(self, optimizer):
        d = deepcopy(self.scheduler.as_dict())
        d["optimizer"] = optimizer
        scheduler = get_scheduler(**d)
        return scheduler

class DataLoaderConfig(ObjectConfig):
    dataset: ObjectConfig

    def build(self):
        dataloader_dict = self.as_dict()
        d = deepcopy(dataloader_dict)
        dataset_args = d.pop("dataset")
        dataset = utils.torch_utils.dict_to_object(dataset_args)
        logger.info(f"Dataset length: {len(dataset)}")
        d["dataset"] = dataset
        dataloader = utils.torch_utils.dict_to_object(d)
        return dataloader

class DirectoryConfig(BaseConfig):
    base: str
    logging: str
    checkpoint: str

class BaseTrainingConfig(BaseConfig):
    accelerator: AcceleratorConfig
    directories: DirectoryConfig
    experiment_name: str
    seed: Optional[int] = None
    total_steps: int
    checkpoint_every_n_steps: int = -1
    validate_every_n_steps: int = -1
    validate_on_checkpoint: bool = True

    def as_dict(self):
        dict_ = super().as_dict()
        dict_['accelerator'] = self.accelerator.as_dict()
        return super().as_dict()
    
    @property
    def experiment_directory(self):
        return os.path.join(self.directories.base, self.experiment_name)
    
    @property
    def logging_directory(self):
        return os.path.join(self.experiment_directory, self.directories.logging)
    
    @property
    def checkpoints_directory(self):
        if self.skip_checkpoints:
            return None
        else:
            return os.path.join(self.experiment_directory, self.directories.checkpoint)
        

    @property
    def skip_checkpoints(self):
        return self.checkpoint_every_n_steps < 0
    
    @property
    def skip_validation(self):
        return self.validate_every_n_steps < 0
