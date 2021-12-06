import torch

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import (
    DataCollatorForLanguageModeling,
    DataCollatorForWholeWordMask,
    BertTokenizer,
    DataCollatorWithPadding
)

from vilt.datasets.memes_classification_dataset import MemesClassificationDataset



def get_pretrained_tokenizer(from_pretrained):
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            BertTokenizer.from_pretrained(
                from_pretrained, do_lower_case="uncased" in from_pretrained
            )
        torch.distributed.barrier()
    return BertTokenizer.from_pretrained(
        from_pretrained, do_lower_case="uncased" in from_pretrained
    )


class MemesClassificationDataModule(LightningDataModule):
    def __init__(self, _config, data_dir='', split_names=['train', 'val']):
        super().__init__()

        self.data_dir = data_dir
        self.split_names = split_names

        self.num_workers = _config["num_workers"]
        self.batch_size = _config["per_gpu_batchsize"]
        self.eval_batch_size = self.batch_size

        self.image_size = 384
        self.max_text_len = 20

        self.train_transform_keys = (
            ["default_train"]
            if len(_config["train_transform_keys"]) == 0
            else _config["train_transform_keys"]
        )

        self.val_transform_keys = (
            ["default_val"]
            if len(_config["val_transform_keys"]) == 0
            else _config["val_transform_keys"]
        )

        tokenizer = _config["tokenizer"]
        self.tokenizer = get_pretrained_tokenizer(tokenizer)
        self.vocab_size = self.tokenizer.vocab_size

        self.mlm_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer
        )
        self.setup_flag = False

    def set_train_dataset(self):
        self.train_dataset = MemesClassificationDataset(
            self.data_dir + self.split_names[0],
            self.train_transform_keys,
            image_size=self.image_size,
            max_text_len=self.max_text_len
        )

    def set_val_dataset(self):
        self.val_dataset = MemesClassificationDataset(
            self.data_dir + self.split_names[1],
            self.val_transform_keys,
            image_size=self.image_size,
            max_text_len=self.max_text_len,
        )

    def setup(self):
        if not self.setup_flag:
            self.set_train_dataset()
            self.set_val_dataset()

            self.train_dataset.tokenizer = self.tokenizer
            self.train_dataset.mlm_collator = self.mlm_collator

            self.val_dataset.tokenizer = self.tokenizer
            self.val_dataset.mlm_collator = self.mlm_collator

            self.setup_flag = True

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset,
            batch_size=128,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.train_dataset.collate,
        )
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.val_dataset,
            batch_size=128,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.val_dataset.collate,
        )
        return loader

