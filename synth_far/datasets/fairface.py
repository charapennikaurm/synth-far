from torch.utils.data import Dataset
import albumentations as A
from typing import Optional, Literal, Union
from albumentations.core.serialization import load as load_albumentations
from ..utils.files import read_json
import torch
from PIL import Image
import numpy as np
from datasets import load_dataset

def center_crop(im, new_sz):
    new_width, new_height = new_sz
    width, height = im.size   # Get dimensions

    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2

    # Crop the center of the image
    im = im.crop((left, top, right, bottom))
    return im

class FairFaceDataset(Dataset):
    def __init__(
        self,
        data_json: Optional[str] = None,
        split: Literal["train", "val", "test"] = "train",
        transforms: Optional[Union[A.Compose, str]] = None,
    ):
        super().__init__()
        assert split in [
            "train",
            "val",
            "test",
        ], f"Split must be one of 'train', 'val', 'test'. But got {split}"
        
        if split == 'test':
            self.data = load_dataset("HuggingFaceM4/FairFace", "1.25")['validation']

            self.indices = list(range(len(self.data)))
            
        else:
            self.data = load_dataset("HuggingFaceM4/FairFace", "1.25")['train']
            self.indices = read_json(data_json)[split]


        if isinstance(transforms, str):
            self.transforms = load_albumentations(transforms)
        else:
            self.transforms = transforms

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        ethnicity2our = {
            0: 2,
            1: 3,
            2: 1,
            3: 0,
            4: 0,
            5: 4,
            6: 2,
        }
        index = self.indices[index]
        image = np.array(center_crop(self.data[index]['image'], (300, 300)))
        age= self.data[index]["age"]
        ethnicity = ethnicity2our[self.data[index]["race"]]
        gender = self.data[index]["gender"]
        
        if self.transforms:
            image = self.transforms(image=image)["image"]
        return image, {
            "age": torch.Tensor([age]).long(),
            "ethnicity": torch.Tensor([ethnicity]).long(),
            "gender": torch.Tensor([gender]).float(),
        }

