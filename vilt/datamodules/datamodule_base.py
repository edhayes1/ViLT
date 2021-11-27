import torch

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import (
    DataCollatorForLanguageModeling,
    DataCollatorForWholeWordMask,
    BertTokenizer,
)

from torch.utils.data.sampler import SubsetRandomSampler


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


class BaseDataModule(LightningDataModule):
    def __init__(self, _config, data_dir=None):
        super().__init__()

        self.data_dir = _config["data_root"] if data_dir is None else data_dir

        self.num_workers = _config["num_workers"]
        self.batch_size = _config["per_gpu_batchsize"]
        self.eval_batch_size = self.batch_size

        self.image_size = _config["image_size"]
        self.max_text_len = _config["max_text_len"]

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

        collator = (
            DataCollatorForWholeWordMask
            if _config["whole_word_masking"]
            else DataCollatorForLanguageModeling
        )

        self.mlm_collator = collator(
            tokenizer=self.tokenizer, mlm=True, mlm_probability=_config["mlm_prob"]
        )
        self.setup_flag = False

    def setup(self, stage):
        if not self.setup_flag:
            self.pretraining_dataset = self.dataset_cls(
                self.data_dir,
                self.train_transform_keys,
                split="train",
                image_size=self.image_size,
                max_text_len=self.max_text_len
            )

            self.train_indices, self.val_indices = torch.utils.data.random_split(list(range(len(self.pretraining_dataset))), 
                                                                        [len(self.pretraining_dataset)-2000, 2000])

            self.pretraining_dataset.tokenizer = self.tokenizer
            self.pretraining_dataset.mlm_collator = self.mlm_collator

            self.setup_flag = True

    def train_dataloader(self):
        loader = DataLoader(
            self.pretraining_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.pretraining_dataset.collate,
            sampler=SubsetRandomSampler(self.train_indices)
        )
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.pretraining_dataset,
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.pretraining_dataset.collate,
            sampler=SubsetRandomSampler(self.val_indices)
        )
        return loader
