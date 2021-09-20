import json
import logging

import azure.functions as func
from .segmentation import segment_image_from_url
import torch
import torchvision

IMG = "img"
CONTAINS_POINT = "contains_point"
MIN_CONFIDENCE = "min_confidence"
MIN_AREA_FRAC = "min_area_frac"
EXTRA_CLASSES = "extra_classes"


def load_parameter_json(req, param, default=None):
    if req.params.get(param) is not None:
        value = json.loads(req.params.get(param))
        logging.info("using a " + param + " filter value of: " + repr(value))
    else:
        logging.info("not using a " + param + " filter")
        value = default
    return value


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')
    image_url = req.params.get(IMG)
    logging.info('Image URL received: ' + image_url)
    logging.info("pytorch version: " + repr(torch.__version__) + " torchvision version: " + repr(torchvision.__version__))
    contains_point = load_parameter_json(req, CONTAINS_POINT)
    min_confidence = load_parameter_json(req, MIN_CONFIDENCE, default=0.2)
    min_area_frac = load_parameter_json(req, MIN_AREA_FRAC, 0.1)
    extra_classes = load_parameter_json(req, EXTRA_CLASSES)

    results = segment_image_from_url(image_url, contains_point=contains_point, min_confidence=min_confidence,
                                     min_area_frac=min_area_frac, include_extra_classes=extra_classes)

    headers = {
        "Content-type": "application/json",
        "Access-Control-Allow-Origin": "*"
    }

    return func.HttpResponse(json.dumps(results, indent=4), headers=headers)

