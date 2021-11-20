from .base_dataset import BaseDataset
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
        image_tensor = self.get_image(id)
        text = self.get_text(id)

        return {
            "image": image_tensor,
            "text": text
        }
