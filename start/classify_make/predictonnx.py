import json
import logging
import os
import sys
import time
from datetime import datetime
from urllib.request import urlopen

import numpy as np  # we're going to use numpy to process input and output data
import onnxruntime  # to inference ONNX models, we use the ONNX Runtime
# display images in notebook
from PIL import Image, ImageOps


def load_labels(path):
    with open(path) as f:
        data = json.load(f)
    return np.asarray(data)

# Run the model on the backend
START_DIR=os.path.dirname(os.path.abspath(__file__))
labelfile=os.path.join(START_DIR , 'labels.json')
model_file = os.path.join(START_DIR, "make_classifier.onnx")

session = onnxruntime.InferenceSession(model_file, None)

# get the name of the first input of the model
input_name = session.get_inputs()[0].name  

labels = load_labels(labelfile)


def preprocess(input_data):
    # convert the input data into the float32 input
    img_data = input_data.astype('float32')

    #normalize
    mean_vec = np.array([0.485, 0.456, 0.406])
    stddev_vec = np.array([0.229, 0.224, 0.225])
    norm_img_data = np.zeros(img_data.shape).astype('float32')
    for i in range(img_data.shape[0]):
        norm_img_data[i,:,:] = (img_data[i,:,:]/255 - mean_vec[i]) / stddev_vec[i]

    #add batch channel
    norm_img_data = norm_img_data.reshape(1,3,240, 240).astype('float32')
    return norm_img_data


def softmax(x):
    x = x.reshape(-1)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def postprocess(result):
    return softmax(np.array(result)).tolist()


def predict_image_from_url(image_url, crop=None):
    """if crop provided it should be a list [center_x,center_y, width, height]. ALl normalised to [0,1]"""
    print("received request!")
    with urlopen(image_url) as testImage:
        image = Image.open(testImage)
    print("Loaded image")

    width, height = image.size
    if crop is not None:
        l, r = crop[0] - crop[2]/2, crop[0] + crop[2]/2
        d, u = crop[1] + crop[3]/2, crop[1] - crop[3]/2
        l, r, d, u = l * width, r * width, d * height, u * height
        imnew = image.crop((l, u, r, d))
        logging.info("cropping image " + repr(image.size) + " to " + repr(imnew.size))

    image_size = 240
    imnew=ImageOps.fit(image, (image_size, image_size))  # loaded as (x,y, 3)
    logging.info("resizing image to: " + repr(imnew.size))

    image_data = np.array(imnew).transpose(2, 0, 1)
    logging.info("image shape: " + repr(image_data.shape))  # (3,x,y)
    input_data = preprocess(image_data)

    start = time.time()
    raw_result = session.run([], {input_name: input_data})
    end = time.time()
    res = postprocess(raw_result)

    inference_time = np.round((end - start) * 1000, 2)
    idx = np.argmax(res)

    response = {
            'created': datetime.utcnow().isoformat(),
            'prediction': labels[idx],
            'inference_time': inference_time,
            'confidence': res[idx],
            'cropped': crop is not None
    }
    logging.info(f'returning {response}')
    return response

if __name__ == '__main__':
    print(predict_image_from_url(sys.argv[1]))
