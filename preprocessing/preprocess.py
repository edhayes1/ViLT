import cv2
import os
import json
import argparse
import logging
import hashlib
from meme_ocr import MemeOCR
import sys
import traceback
import torch
import numpy as np

logfile = 'preprocess.log'
logging.basicConfig(filename=logfile, level=logging.INFO)

parser = argparse.ArgumentParser(description='Get OCR and object boxes from input memes')
parser.add_argument('-i', '--input', type=str, help='Input file folder', required=True)
parser.add_argument('-o', '--output', type=str, help='Output file folder', required=True)
parser.add_argument('--fill', dest='fill', action='store_true', help="Fill text boxes")
parser.add_argument("--max_num_text_areas", type=int, default=5,
                    help="maximum number of text areas allowed in the image, otherwise discard")

args = parser.parse_args()

ocr = MemeOCR(args.input, args.fill)

def log_except_hook(*exc_info):
    text = "".join(traceback.format_exception(*exc_info))
    logging.fatal("Unhandled exception: %s", text)


sys.excepthook = log_except_hook

num_processed = 0
removed = 0
for filename in os.listdir(args.input):
    num_processed += 1
    input_path = args.input + filename
    output_path = args.output + filename
    try:
        image = cv2.imread(input_path)
        if image is None or image.shape[0] < 50 or image.shape[1] < 50:
            logging.error('Cannot read file from path: {}, or too small, deleting...'.format(input_path))
            os.remove(input_path)
            continue

        with torch.no_grad():
            image, text_data = ocr(filename, image)
    except:
        logging.error('Cannot read file from path: {}, deleting...'.format(input_path))
        os.remove(input_path)
        removed += 1
        continue

    if len(text_data) == 0 \
        or len(text_data) > args.max_num_text_areas \
        or len(" ".join([t['text'] for t in text_data]).split()) < 8:
        os.remove(input_path)
        removed += 1
        logging.info('Filtered file {}'.format(filename))
        continue

    data = {}
    data['id'] = filename
    data['size'] = [image.shape[0], image.shape[1]]
    data['text_data'] = text_data

    with open(output_path + '.json', 'w') as f:
        json.dump(data, f)

    cv2.imwrite(output_path, image)

    logging.info('Saved file {} to {}'.format(filename, output_path))

logging.info('Filtered {} images, Completed {} images'.format(removed, num_processed - removed))