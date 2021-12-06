import random
import torch
import io
import pyarrow as pa
import os
import json
import numpy as np

from PIL import Image
from vilt.transforms.pixelbert import pixelbert_transform, precomputed_transform


class BaseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir: str,
        transform_keys: list,
        image_size: int,
        max_text_len=20,
        split='train'
    ):
        """
        data_dir : where dataset file *.arrow lives; existence should be guaranteed via DataModule.prepare_data
        transform_keys : keys for generating augmented views of images
        text_column_name : pyarrow table column name that has list of strings as elements
        """
        assert len(transform_keys) >= 1
        super().__init__()

        self.transforms = precomputed_transform()

        self.image_size = image_size
        self.max_text_len = max_text_len
        self.data_dir = data_dir

        self.table = pa.ipc.RecordBatchFileReader(
            pa.memory_map(f"{data_dir}/pretrain_{split}_192.arrow", "r")).read_all()

        if split == 'train':
            hateful_memes_table = pa.ipc.RecordBatchFileReader(
                pa.memory_map(f"{data_dir}/pretrain_hate_192.arrow", "r")).read_all()

            self.table = pa.concat_tables([self.table, hateful_memes_table], promote=True)

        self.all_texts = self.table['texts']
        
    def __len__(self):
        return len(self.all_texts)

    def get_image(self, index):
        image_bytes = io.BytesIO(self.table['image'][index].as_py())
        image_bytes.seek(0)
        image = Image.open(image_bytes).convert("RGB")
        image = np.array(image)
        image_0 = self.transforms(image=image)["image"]
        image_1 = self.transforms(image=image)["image"]
        # return torch.rand_like(image_0), torch.rand_like(image_1)

        return image_0, image_1
    
    def split_text(self, text):
        text = text.split()
        if len(text) <= 2:
            t1, t2 = " ".join(text), ""
        else:
            random_split = random.randrange(1, len(text)-1)
            t1 = " ".join(text[:random_split])
            t2 = " ".join(text[random_split:])

        return t1, t2
    
    def tokenise(self, text):
        return self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.max_text_len,
                return_special_tokens_mask=True,
            )

    def get_text(self, id, split=True):

        text = " ".join(self.all_texts[id].as_py())
        if not split:
            e = self.tokenise(text)
            return (text, e)

        t1, t2 = self.split_text(text)
        e1 = self.tokenise(t1)
        e2 = self.tokenise(t2)
        
        return [(t1, e1), (t2, e2)]

    def collate(self, batch):
        batch_size = len(batch)
        keys = set([key for b in batch for key in b.keys()])
        batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}
        ret = {}

        img_keys = ['image_0', 'image_1']
        img_sizes = list()

        for img_key in img_keys:
            img = batch[img_key]
            img_sizes += [i.shape for i in img if i is not None]

        for size in img_sizes:
            assert (
                len(size) == 3
            ), f"Collate error, an image should be in shape of (3, H, W), instead of given {size}"

        if len(img_keys) != 0:
            max_height = max([i[1] for i in img_sizes])
            max_width = max([i[2] for i in img_sizes])

        for img_key in img_keys:
            img = batch[img_key]

            new_images = [
                torch.zeros(batch_size, 3, max_height, max_width)
            ]

            for bi in range(batch_size):
                orig = img[bi]
                new_images[0][bi, :, : orig.shape[1], : orig.shape[2]] = orig

            ret[img_key] = new_images

        txt_keys = ['text_0', 'text_1']

        if len(txt_keys) != 0:
            # texts = [[d[0] for d in dict_batch[txt_key]] for txt_key in txt_keys]
            encodings = [[d[1] for d in batch[txt_key]] for txt_key in txt_keys]
            flatten_encodings = [e for encoding in encodings for e in encoding]
            flatten_mlms = self.mlm_collator(flatten_encodings)

            for i, txt_key in enumerate(txt_keys):
                texts, encodings = (
                    [d[0] for d in batch[txt_key]],
                    [d[1] for d in batch[txt_key]],
                )

                input_ids, mlm_labels = (
                    flatten_mlms["input_ids"][batch_size * (i) : batch_size * (i + 1)],
                    flatten_mlms["labels"][batch_size * (i) : batch_size * (i + 1)],
                )

                attention_mask = torch.zeros_like(input_ids)
                for _i, encoding in enumerate(encodings):
                    _attention_mask = torch.tensor(encoding["attention_mask"])
                    attention_mask[_i, : len(_attention_mask)] = _attention_mask

                # dict_batch[txt_key] = texts
                ret[f"{txt_key}_labels"] = torch.full_like(input_ids, -100)
                ret[f"{txt_key}_ids_mlm"] = input_ids
                ret[f"{txt_key}_labels_mlm"] = mlm_labels
                ret[f"{txt_key}_masks"] = attention_mask

        return ret
