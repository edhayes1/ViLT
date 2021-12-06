import cv2
import os
import numpy as np
import json
import argparse
import logging
import hashlib
from meme_ocr import MemeOCR
import sys
import traceback
import torch
import pandas as pd

logfile = 'preprocess.log'
logging.basicConfig(filename=logfile, level=logging.INFO)


def get_data(data_dir):
    df = pd.read_csv(data_dir + 'final_dank.csv')
    df = df[['id', 'dank_level']]
    df = df.rename(columns={"dank_level": "label"})
    df.id =df.id.astype(str)
    train = [filename[:-4] for filename in os.listdir(data_dir + 'train') if '(1)' not in filename]
    test = [filename[:-4] for filename in os.listdir(data_dir + 'test') if '(1)' not in filename]

    train = pd.DataFrame({'id': train})    
    test = pd.DataFrame({'id': test})
    train.id = train.id.astype(str)
    test.id = test.id.astype(str)

    train = train.merge(df, on='id')
    test = test.merge(df, on='id')

    return train, test


parser = argparse.ArgumentParser(description='Get OCR and object boxes from input memes')
parser.add_argument('-i', '--input', type=str, help='Input file folder', required=True)
parser.add_argument('-o', '--output', type=str, help='Output file folder', required=True)
parser.add_argument('--fill', dest='fill', action='store_true', help="Fill text boxes")

args = parser.parse_args()

ocr = MemeOCR(args.input, args.fill)


def log_except_hook(*exc_info):
    text = "".join(traceback.format_exception(*exc_info))
    logging.fatal("Unhandled exception: %s", text)


sys.excepthook = log_except_hook
data = get_data(args.input)
# data = get_data('/data/edward/dank_data/original_data/')
logging.info("{} Data points".format(len(data)))

removed = 0
num_processed = 0
for split, data in enumerate(data):
    for _, row in data.iterrows():
        training_dir = 'train/' if split == 0 else 'test/'
        id = row['id']
        filename = id + '.jpg'
        label = row['label']

        num_processed += 1
        input_path = args.input + training_dir + filename
        output_path = args.output + training_dir + filename
        try:
            image = cv2.imread(input_path)
            if image is None:
                logging.error('Cannot read file from path: {}, continuing...'.format(input_path))
                continue

            image, text_data = ocr(filename, image)
        except:
            logging.error('Cannot read file from path: {}, continuing...'.format(input_path))
            continue

        if len(text_data) == 0:
            removed += 1
            logging.info('Filtered file {}'.format(filename))
            continue

        d = {}
        d['id'] = filename
        d['size'] = [image.shape[0], image.shape[1]]
        d['label'] = label
        d['text_data'] = text_data

        with open(output_path + '.json', 'w') as f:
            json.dump(d, f)

        cv2.imwrite(output_path, image)

        logging.info('Saved file {} to {}'.format(filename, output_path))

logging.info('Filtered {} images, Completed {} images'.format(removed, num_processed - removed))