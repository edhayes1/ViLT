import random
import torch
import io
import pyarrow as pa
import os
import json

from PIL import Image
from vilt.transforms.pixelbert import pixelbert_transform


class BaseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir: str,
        transform_keys: list,
        image_size: int,
        max_text_len=20,
        img_dir: str = '/data/edward/images/dir_001/'
    ):
        """
        data_dir : where dataset file *.arrow lives; existence should be guaranteed via DataModule.prepare_data
        transform_keys : keys for generating augmented views of images
        text_column_name : pyarrow table column name that has list of strings as elements
        """
        assert len(transform_keys) >= 1
        super().__init__()

        self.transforms = pixelbert_transform(image_size)

        self.image_size = image_size
        self.max_text_len = max_text_len
        self.data_dir = data_dir
        self.img_dir = img_dir

        self.instances, self.all_texts = self.read_data(self.img_dir, self.data_dir)
    
    def read_data(self, img_dir, data_dir):
        instances = []
        text_data = {}

        img_dir = os.path.expanduser(img_dir)
        for root, _, fnames in os.walk(img_dir, followlinks=True):
            for fname in fnames:
                instances.append(fname)
                if len(instances) > 500:
                    break

        data_dir = os.path.expanduser(data_dir)
        for root, _, fnames in os.walk(data_dir, followlinks=True):
            for fname in fnames:
                if fname.lower().endswith('json'):
                    id = fname[:-5]  # remove extension to get it's unique ID
                    with open(data_dir + fname) as f:
                        d = {}
                        data = json.load(f)

                        if id in instances:
                            d['text_data'] = " ".join(entry['text'] for entry in data['text_data'])

                            if 'src_transcript' in data:
                                d['text_data'] = data['src_transcript']
                            
                            if 'labels' in data:
                                d['labels'] = data['labels']

                            text_data[id] = d

        return instances, text_data

    @property
    def corpus(self):
        return [text for texts in self.all_texts for text in texts]

    def __len__(self):
        return len(self.instances)

    def get_image(self, id, views=True):
        path = self.img_dir + id
        image = Image.open(path).convert("RGB")

        if views:
            return self.transforms(image), self.transforms(image)

        return self.transforms(image)
    
    def split_text(self, text):
        text = text.split()
        if len(text) <= 2:
            t1, t2 = text, ""
        else:
            random_split = random.randrange(1, len(text)-1)
            t1 = [" ".join(text[:random_split])]
            t2 = [" ".join(text[random_split:])]

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

        text = self.all_texts[id]
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
        dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}

        img_keys = [k for k in list(dict_batch.keys()) if "image" in k]
        img_sizes = list()

        for img_key in img_keys:
            img = dict_batch[img_key]
            img_sizes += [i.shape for i in img if i is not None]

        for size in img_sizes:
            assert (
                len(size) == 3
            ), f"Collate error, an image should be in shape of (3, H, W), instead of given {size}"

        if len(img_keys) != 0:
            max_height = max([i[1] for i in img_sizes])
            max_width = max([i[2] for i in img_sizes])

        for img_key in img_keys:
            img = dict_batch[img_key]

            new_images = [
                torch.zeros(batch_size, 3, max_height, max_width)
            ]

            for bi in range(batch_size):
                orig = img[bi]
                new_images[0][bi, :, : orig.shape[1], : orig.shape[2]] = orig

            dict_batch[img_key] = new_images

        txt_keys = [k for k in list(dict_batch.keys()) if "text" in k]

        if len(txt_keys) != 0:
            texts = [[d[0] for d in dict_batch[txt_key]] for txt_key in txt_keys]
            encodings = [[d[1] for d in dict_batch[txt_key]] for txt_key in txt_keys]
            draw_text_len = len(encodings)
            flatten_encodings = [e for encoding in encodings for e in encoding]
            flatten_mlms = self.mlm_collator(flatten_encodings)

            for i, txt_key in enumerate(txt_keys):
                texts, encodings = (
                    [d[0] for d in dict_batch[txt_key]],
                    [d[1] for d in dict_batch[txt_key]],
                )

                mlm_ids, mlm_labels = (
                    flatten_mlms["input_ids"][batch_size * (i) : batch_size * (i + 1)],
                    flatten_mlms["labels"][batch_size * (i) : batch_size * (i + 1)],
                )

                input_ids = torch.zeros_like(mlm_ids)
                attention_mask = torch.zeros_like(mlm_ids)
                for _i, encoding in enumerate(encodings):
                    _input_ids, _attention_mask = (
                        torch.tensor(encoding["input_ids"]),
                        torch.tensor(encoding["attention_mask"]),
                    )
                    input_ids[_i, : len(_input_ids)] = _input_ids
                    attention_mask[_i, : len(_attention_mask)] = _attention_mask

                dict_batch[txt_key] = texts
                dict_batch[f"{txt_key}_ids"] = input_ids
                dict_batch[f"{txt_key}_labels"] = torch.full_like(input_ids, -100)
                dict_batch[f"{txt_key}_ids_mlm"] = mlm_ids
                dict_batch[f"{txt_key}_labels_mlm"] = mlm_labels
                dict_batch[f"{txt_key}_masks"] = attention_mask

        return dict_batch
