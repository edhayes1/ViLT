from vilt.datasets import MemesDataset
from .datamodule_base import BaseDataModule


class MemesDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return MemesDataset

    @property
    def dataset_name(self):
        return "memes"
