from .base_dataset import BaseDataset
import torch
import sys
import random


class MemesDataset(BaseDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split

        super().__init__(
            *args,
            **kwargs
        )

    def __getitem__(self, index):

        id = self.instances[index]
        image_tensor_0, image_tensor_1 = self.get_image(id)
        text_0, text_1 = self.get_text(id)

        return {
            "image_0": image_tensor_0,
            "image_1": image_tensor_1,
            "text_0": text_0,
            "text_1": text_1
        }