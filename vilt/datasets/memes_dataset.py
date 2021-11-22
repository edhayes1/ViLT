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


class HatefulMemesDataset(BaseDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val"]
        self.split = split

        super().__init__(
            *args,
            **kwargs,
            img_dir='/data/edward/hateful/original_data/img/' + split
        )
    
    def __getitem__(self, index):

        id = self.instances[index]
        image_tensor = self.get_image(id, views=False)
        text = self.get_text(id, split=False)
        label = torch.FloatTensor(self.text_data[id]['labels'])

        return {
            "image": image_tensor,
            "text": text,
            "label": label
        }

class FeaturesDataset(BaseDataset):
    '''
        Dataset of features for batch training of a linear classifer
    '''
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        return (self.features[idx], self.labels[idx])