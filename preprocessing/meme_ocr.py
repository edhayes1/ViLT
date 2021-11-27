from easyocr import Reader
import cv2
import numpy as np

import logging
logger = logging.getLogger(__name__)


class MemeOCR:
    def __init__(self, input_folder, fill=True):
        self.input_folder = input_folder
        self.fill = fill
        self.reader = Reader(['en'])

    def _cleanup_text(self, text):
        # strip out non-ASCII text so we can draw the text on the image
        # using OpenCV
        return "".join([c if ord(c) < 128 else "" for c in text]).strip()

    def __call__(self, file_id, image):
        results = self.reader.readtext(image, paragraph=True, decoder='beamsearch', batch_size=32)  # height_ths=5.0)

        text_boxes = []
        for (bbox, text) in results:
            text_box = {}
            (tl, tr, br, bl) = bbox
            tl = (int(tl[0]), int(tl[1]))
            tr = (int(tr[0]), int(tr[1]))
            br = (int(br[0]), int(br[1]))
            bl = (int(bl[0]), int(bl[1]))

            if self.fill:
                points = np.array([list(tl), list(tr), list(br), list(bl)], dtype=np.int32)
                cv2.fillPoly(image, [points], (0, 0, 0))

            text = self._cleanup_text(text)

            text_box['tl'] = tl
            text_box['br'] = br
            text_box['area'] = (br[0] - tl[0]) * (br[1] - tl[1])
            text_box['text'] = text
            text_boxes.append(text_box)

        logger.info('Found {} text regions in file: {}'.format(len(text_boxes), file_id))
        return image, text_boxes
