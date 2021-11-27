import json
import pandas as pd
import pyarrow as pa
import random
import os

from tqdm import tqdm
from vilt.transforms.utils import (
    inception_normalize,
    MinMaxResize,
)
from torchvision import transforms


def pixelbert_transform(size=384):
    longer = int((1333 / 800) * size)
    return MinMaxResize(shorter=size, longer=longer)


def path2rest(id, data):
    path = data['path']
    texts = data['text_data']

    with open(path, "rb") as fp:
        binary = fp.read()

    return [binary, texts, id]


def read_data(data_root):
    text_data = {}

    data_dir = os.path.expanduser(data_root)
    for root, _, fnames in os.walk(data_dir, followlinks=True):
        for fname in fnames:
            if fname.lower().endswith('json'):
                id = fname[:-5]
                with open(data_dir + fname) as f:
                    d = {}
                    data = json.load(f)
                    d['text_data'] = [entry['text'] for entry in data['text_data']]
                    d['path'] = data_root + id

                    if 'src_transcript' in data:
                        d['text_data'] = data['src_transcript']
                    
                    if 'label' in data:
                        d['label'] = data['label']

                    text_data[id] = d
                        

    return text_data


def make_arrow(root, dataset_root):
    data = read_data(root)

    bs = [path2rest(id, data) for id, data in tqdm(data.items())]

    dataframe = pd.DataFrame(
        bs, columns=["image", "texts", "image_id"],
    )

    table = pa.Table.from_pandas(dataframe)

    os.makedirs(dataset_root, exist_ok=True)
    with pa.OSFile(
        f"{dataset_root}/pretrain.arrow", "wb"
    ) as sink:
        with pa.RecordBatchFileWriter(sink, table.schema) as writer:
            writer.write_table(table)

make_arrow('/data/edward/pretrain/', '/data/edward/')