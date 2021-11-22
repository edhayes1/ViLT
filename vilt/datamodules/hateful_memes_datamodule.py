from vilt.datasets import MemesDataset
from vilt.datasets.memes_dataset import HatefulMemesDataset
from .datamodule_base import BaseDataModule


class HatefulMemesDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return HatefulMemesDataset

    @property
    def dataset_name(self):
        return "hateful memes"
