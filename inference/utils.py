import cv2
import time
import copy
import logging
import numpy as np
from PIL import Image


formatter = logging.Formatter('%(asctime)s.%(msecs)d %(message)s', '%Y-%m-%d %H:%M:%S')

def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""

    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


def log_and_print(logger, level, message):
    """Log and print a message."""
    print(message)
    if level == "info":
        logger.info(message)
    elif level == "debug":
        logger.debug(message)
    elif level == "warning":
        logger.warning(message)
    elif level == "error":
        logger.error(message)
    elif level == "critical":
        logger.critical(message)
        
def load_capture(args):
    capture = cv2.VideoCapture(args.input)
    return capture

def make_palette(num_classes):
    """
    Maps classes to colors in the style of PASCAL VOC.
    Close values are mapped to far colors for segmentation visualization.
    See http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit


    Takes:
        num_classes: the number of classes
    Gives:
        palette: the colormap as a k x 3 array of RGB colors
    """
    palette = np.zeros((num_classes, 3), dtype=np.uint8)
    for k in range(0, num_classes):
        label = k
        i = 0
        while label:
            palette[k, 0] |= (((label >> 0) & 1) << (7 - i))
            palette[k, 1] |= (((label >> 1) & 1) << (7 - i))
            palette[k, 2] |= (((label >> 2) & 1) << (7 - i))
            label >>= 3
            i += 1
    return palette

def draw_action_label(img, palette, action_dict, label):
    fix_buffer = 12
    for i in range(len(label)):
        k = label[i]
        color_plate = (int(palette[k][2]), int(palette[k][1]), int(palette[k][0]))
        img = cv2.rectangle(img, (5, 15 + fix_buffer * i), (25, 5 + fix_buffer * i), color_plate, thickness=-1)
        cv2.putText(img, action_dict[k], (30, 12 + fix_buffer * i), cv2.FONT_HERSHEY_COMPLEX, 0.25, color_plate, 1)
        
    return img

def label_arr2img(label_queue, palette):
    data = list(copy.deepcopy(label_queue.queue))
    array = np.array(data).transpose()
    arr = array.astype(np.uint8)
    arr = np.tile(arr, (20, 1))
    img = Image.fromarray(arr)
    img = img.convert("P")
    img.putpalette(palette)
    return img
