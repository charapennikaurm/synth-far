from torch.utils.data import Dataset
import albumentations as A
from typing import Optional, Literal, Union
from albumentations.core.serialization import load as load_albumentations
from ..utils.files import read_json
import torch
from PIL import Image
import numpy as np

class SynthFARDataset(Dataset):
    def __init__(
        self,
        data_json: str,
        split: Literal["train", "val", "test"] = "train",
        transforms: Optional[Union[A.Compose, str]] = None,
    ):
        super().__init__()
        assert split in [
            "train",
            "val",
            "test",
        ], f"Split must be one of 'train', 'val', 'test'. But got {split}"

        self.data_json = read_json(data_json)

        self.data_json = [
            k for k in self.data_json if k["split"] == split
        ]


        if isinstance(transforms, str):
            self.transforms = load_albumentations(transforms)
        else:
            self.transforms = transforms

    def __read_image(self, image_path: str):
        image = Image.open(image_path)
        image = image.convert("RGB")
        return np.array(image)

    def __len__(self):
        return len(self.data_json)

    def __getitem__(self, index):
        image = self.__read_image(self.data_json[index]["image_path"])
        age= self.data_json[index]["age"]
        ethnicity = self.data_json[index]["ethnicity"]
        gender = self.data_json[index]["gender"]
        if self.transforms:
            image = self.transforms(image=image)["image"]
        return image, {
            "age": torch.Tensor([age]).float(),
            "ethnicity": torch.Tensor([ethnicity]).long(),
            "gender": torch.Tensor([gender]).float(),
        }

