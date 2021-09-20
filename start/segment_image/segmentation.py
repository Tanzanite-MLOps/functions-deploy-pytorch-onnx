import logging
import os
import sys
import time
from datetime import datetime
from urllib.request import urlopen

import numpy as np  # we're going to use numpy to process input and output data
# display images in notebook
from PIL import Image, ImageOps

from .yolo.yolo import get_vehicles_boxes

START_DIR=os.path.dirname(os.path.abspath(__file__))


def segment_image_from_url(image_url, **kwargs):
    logging.info("received segmentation request!")
    with urlopen(image_url) as testImage:
        image = Image.open(testImage)
    logging.info("Loaded image")

    imnew=ImageOps.fit(image, (240, 240))

    image_data = np.array(imnew).transpose(2, 0, 1)
    logging.info("image shape: " + repr(image_data.shape)) # (3,x,y)
    input_data = image_data.astype('float32')

    start = time.time()
    vehicles = get_vehicles_boxes(input_data, **kwargs)
    end = time.time()

    inference_time = np.round((end - start) * 1000, 2)

    response = {
            'created': datetime.utcnow().isoformat(),
            'inference_time': inference_time,
            'vehicles': vehicles
    }
    logging.info(f'returning {response}')
    return response


if __name__ == '__main__':
    print(segment_image_from_url(sys.argv[1]))
