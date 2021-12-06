import json
import pandas as pd
import pyarrow as pa
import random
import os
from PIL import Image
import io

from tqdm import tqdm
from vilt.transforms.utils import (
    inception_normalize,
    MinMaxResize,
)
from torchvision import transforms
from multiprocessing import Pool


def pixelbert_transform(size=192):
    longer = int((1333 / 800) * size)
    return MinMaxResize(shorter=size, longer=longer)

transform = pixelbert_transform()

def multi_run_wrapper(args):
   return path2rest(*args)


def path2rest(id, data):
    path = data['path']
    texts = data['text_data']

    img = Image.open(path).convert("RGB")
    try:
        img = transform(img)
    except:
        # os.remove(path)
        # os.remove(path+'.json')
        print(path)
        return [None, None, None]
    
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='jpeg')
    img_byte_arr = img_byte_arr.getvalue()

    return [img_byte_arr, texts, id]


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
                    d['path'] = data_root + id

                    if 'src_transcript' in data:
                        d['text_data'] = [data['src_transcript']]
                    else:
                        d['text_data'] = [entry['text'] for entry in data['text_data']]
                    
                    # if 'label' in data:
                    #     d['label'] = data['label']

                    text_data[id] = d

    return text_data

def write(dataset_root, dataframe, out_path):
    table = pa.Table.from_pandas(dataframe)

    os.makedirs(dataset_root, exist_ok=True)
    with pa.OSFile(out_path, "wb") as sink:
        with pa.RecordBatchFileWriter(sink, table.schema) as writer:
            writer.write_table(table)


def make_arrow(root, dataset_root, split=False):
    data = read_data(root)

    with Pool(62) as pool:
        bs = pool.map(multi_run_wrapper, [(id, data) for id, data in tqdm(data.items())])

    dataframe = pd.DataFrame(
        [b for b in bs if b[0] is not None], columns=["image", "texts", "image_id"],
    )

    if split:
        train_data = dataframe.head(n=len(dataframe) - 4000)
        val_data = dataframe.tail(n=4000)

        print(f'len train:{len(train_data)}')
        print(f'len val:{len(val_data)}')

        write(dataset_root, train_data, f"{dataset_root}/pretrain_train_192.arrow")
        write(dataset_root, val_data, f"{dataset_root}/pretrain_val_192.arrow")
    else:
        write(dataset_root, dataframe, f"{dataset_root}/pretrain_train_192.arrow")



make_arrow('/data/edward/pretrain/', '/data/edward/', split=True)