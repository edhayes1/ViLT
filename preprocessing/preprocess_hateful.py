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

logfile = 'preprocess.log'
logging.basicConfig(filename=logfile, level=logging.INFO)


def get_data(data_dir):
    data = []

    with open(data_dir + 'train.jsonl') as f:
        for line in f:
            d = json.loads(line)
            d['training'] = True
            data.append(d)

    with open(data_dir + 'dev_seen.jsonl') as f:
        for line in f:
            d = json.loads(line)
            d['training'] = True
            data.append(d)

    with open(data_dir + 'dev_unseen.jsonl') as f:
        for line in f:
            d = json.loads(line)
            d['training'] = True
            data.append(d)

    with open(data_dir + 'test_seen.jsonl') as f:
        for line in f:
            d = json.loads(line)
            d['training'] = True
            data.append(d)

    with open(data_dir + 'test_unseen.jsonl') as f:
        for line in f:
            d = json.loads(line)
            d['training'] = False
            data.append(d)

    return data


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
logging.info("{} Data points".format(len(data)))

removed = 0
num_processed = 0
for data_dict in data:
    filename = data_dict['img']
    label = data_dict['label']
    pre_extracted_text = data_dict['text']
    training = data_dict['training']
    training_dir = 'vilt_train/' if training else 'vilt_test/'

    num_processed += 1
    input_path = args.input + filename
    output_path = args.output + training_dir + filename[4:] # remove "img/"
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
    d['training'] = training
    d['src_transcript'] = pre_extracted_text

    with open(output_path + '.json', 'w') as f:
        json.dump(d, f)

    cv2.imwrite(output_path, image)

    logging.info('Saved file {} to {}'.format(filename, output_path))

logging.info('Filtered {} images, Completed {} images'.format(removed, num_processed - removed))