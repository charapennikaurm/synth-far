from torch.utils.data import Dataset
import albumentations as A
from typing import Optional, Literal, Union
from albumentations.core.serialization import load as load_albumentations
from synth_far.utils.files import read_json
import torch
from PIL import Image
import numpy as np
from datasets import load_dataset

class UTKDataset(Dataset):
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

        self.data = load_dataset("nu-delta/utkface")['train']
        
        self.indices = read_json(data_json)[split]


        if isinstance(transforms, str):
            self.transforms = load_albumentations(transforms)
        else:
            self.transforms = transforms

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        gender2id = {
            "Male": 0,
            "Female": 1,
        }
        ethnicity2id = {
            "White": 0,
            "Black": 1,
            "Asian": 2,
            "Indian": 3,
            "Other": 4,
        }
        index = self.indices[index]
        image = np.array(self.data[index]['image'])
        age= self.data[index]["age"]
        ethnicity = ethnicity2id[self.data[index]["ethnicity"]]
        gender = gender2id[self.data[index]["gender"]]
        
        if self.transforms:
            image = self.transforms(image=image)["image"]
        return image, {
            "age": torch.Tensor([age]).float(),
            "ethnicity": torch.Tensor([ethnicity]).long(),
            "gender": torch.Tensor([gender]).float(),
        }


