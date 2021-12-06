import random
import torch
import io
import pyarrow as pa
import os
import json

from PIL import Image
from vilt.transforms.pixelbert import pixelbert_transform, precomputed_transform



class MemesClassificationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir: str,
        transform_keys: list,
        image_size: int,
        max_text_len=40
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

        self.instances, self.all_texts = self.read_data()
    
    def read_data(self):
        instances = []
        text_data = {}

        data_dir = os.path.expanduser(self.data_dir)
        for root, _, fnames in os.walk(data_dir, followlinks=True):
            for fname in fnames:
                if fname.lower().endswith('json'):
                    id = fname[:-5]
                    instances.append(id)
                    with open(data_dir + fname) as f:
                        d = {}
                        data = json.load(f)
                        if 'text_data' in data:
                            d['text_data'] = " ".join(entry['text'] for entry in data['text_data'])

                        elif 'src_transcript' in data:
                            d['text_data'] = data['src_transcript']
                        
                        if 'label' in data:
                            d['label'] = data['label']

                        text_data[id] = d

        return instances, text_data

    @property
    def corpus(self):
        return [text for texts in self.all_texts for text in texts]

    def __len__(self):
        return len(self.instances)

    def get_image(self, id):
        path = self.data_dir + id
        image = Image.open(path).convert("RGB")

        return self.transforms(image)
    
    def tokenise(self, text):
        return self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.max_text_len,
                return_special_tokens_mask=True,
            )

    def get_text(self, id):
        text = self.all_texts[id]['text_data']
        e = self.tokenise(text)
        return (text, e)


    def collate(self, batch):
        batch_size = len(batch)
        keys = set([key for b in batch for key in b.keys()])
        dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}

        img_sizes = list()

        img = dict_batch['image']
        img_sizes += [i.shape for i in img if i is not None]

        for size in img_sizes:
            assert (
                len(size) == 3
            ), f"Collate error, an image should be in shape of (3, H, W), instead of given {size}"

        max_height = max([i[1] for i in img_sizes])
        max_width = max([i[2] for i in img_sizes])

        img = dict_batch['image']
        new_images = torch.zeros(batch_size, 3, max_height, max_width)
        

        for bi in range(batch_size):
            orig = img[bi]
            new_images[bi, :, : orig.shape[1], : orig.shape[2]] = orig

        dict_batch['image'] = new_images

        texts = [d[0] for d in dict_batch['text']]
        encodings = [d[1] for d in dict_batch['text']]
        flatten_encodings = {k: [dic[k] for dic in encodings] for k in encodings[0]}
        input_ids = torch.tensor(flatten_encodings['input_ids'])
        attention_mask = torch.tensor(flatten_encodings["attention_mask"])

        if 'label' in dict_batch:
            dict_batch['label'] = torch.FloatTensor(dict_batch['label'])

        dict_batch['text'] = texts
        dict_batch["text_ids"] = input_ids
        dict_batch["text_masks"] = attention_mask

        return dict_batch

    def __getitem__(self, index):

        id = self.instances[index]
        image_tensor = self.get_image(id)
        text = self.get_text(id)
        label = self.all_texts[id]['label']

        return {
            "image": image_tensor,
            "text": text,
            "label": label
        }


class FeaturesDataset(torch.utils.data.Dataset):
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