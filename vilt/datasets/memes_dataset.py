from .base_dataset import BaseDataset
import torch
import sys
import random


class MemesDataset(BaseDataset):
    def __init__(self, *args, **kwargs):

        super().__init__(
            *args,
            **kwargs
        )

    def __getitem__(self, index):
        image_tensor_0, image_tensor_1 = self.get_image(index)
        text_0, text_1 = self.get_text(index)

        return {
            "image_0": image_tensor_0,
            "image_1": image_tensor_1,
            "text_0": text_0,
            "text_1": text_1
        }