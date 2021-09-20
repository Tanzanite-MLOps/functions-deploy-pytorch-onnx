import json
import logging

import azure.functions as func
from .predictonnx import predict_image_from_url

IMG = "img"
CROP = "crop"

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')
    image_url = req.params.get(IMG)
    if req.params.get(CROP) is not None:
        crop = json.loads(req.params.get(CROP))
        logging.info("got crop arg: " + repr(crop) + " expected in format [center_x,  center_y, width, height] "
                                            "with all values normalised to [0,1], no spaces")
    else:
        logging.info("no crop args")
        crop = None

    logging.info('Image URL received: ' + image_url)

    results = predict_image_from_url(image_url, crop=crop)

    headers = {
        "Content-type": "application/json",
        "Access-Control-Allow-Origin": "*"
    }

    return func.HttpResponse(json.dumps(results, indent=4), headers=headers)

