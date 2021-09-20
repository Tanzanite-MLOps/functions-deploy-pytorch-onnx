import json
import logging
from typing import Tuple

import torch

logging.info('loading yolo model')
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
logging.info('loaded yolo model')

logging.info("all yolo names: " + repr(model.names))


def segment_image_from_url(url):
    """for local testing only. not azure"""
    results = model(url)
    results.save()
    vehicle_detections = filter_vehicle_results(results, contains_point=(0.5, 0.5), include_extra_classes=["bus"])
    print("vehicles:\n", vehicle_detections)


def get_vehicles_boxes(image, **kwargs):
    results = model(image)
    vehicle_detections = filter_vehicle_results(results, **kwargs)
    return vehicle_detections


def filter_vehicle_results(results, min_area_frac=None, min_confidence=None,
                           contains_point: Tuple[float, float] = None, include_extra_classes=None):
    """
        gets all the vehicles detected, with additional filters
        :parameter min_area_frac ~ [0,1] - minimum area a vehicle can ocupy before being filtered out as background
        :parameter max_distance - the furthest the center of the
        :parameter contains_point ~ tuple(x,y) - if provided, only the vehicle boxes which contain the given point will be returned
        :parameter include_extra_classes - a list of extra classes, eg ["bus"] to be included in the filter
    """
    vehicle_detections = []  # number of batches
    logging.info("got results: " + repr(results))

    for detection in results.pandas().xywhn:
        name_filter = (detection['name'] == "car") | (detection['name'] == "truck") | (detection['name'] == "bus")
        if include_extra_classes is not None:
            for extra in include_extra_classes:
                name_filter |= detection['name'] == extra
                logging.info("allowing class: " + repr(extra) + " through filter")

        optional_filters = detection['xcenter'] > 0

        if min_area_frac is not None:
            x = detection["xcenter"]
            y = detection["ycenter"]
            w = detection["width"]
            h = detection["height"]
            detection["area"] = w * h

            optional_filters &= detection['area'] > min_area_frac
            logging.info("appying the min area frac " + repr(min_area_frac) + " filter")

        if min_confidence is not None:
            optional_filters &= detection['confidence'] > min_confidence
            logging.info("appying the min confidence " + repr(min_confidence) + " filter")

        if contains_point is not None:
            """must check if the point is contained in the bounding box"""
            contains_x = (x - w / 2 < contains_point[0]) & (contains_point[0] < x + w / 2)
            contains_y = (y - h / 2 < contains_point[1]) & (contains_point[1] < y + h / 2)
            detection["contains_point"] = contains_x & contains_y

            optional_filters &= detection["contains_point"] == True
            logging.info("appying the contains point " + repr(contains_point) + " filter")

        vehicles = detection.loc[optional_filters & name_filter]
        vehicle_detections.append(json.loads(vehicles.to_json()))
    return vehicle_detections


if __name__ == "__main__":
    # 'https://ultralytics.com/images/zidane.jpg'
    # "https://www.msi.org.za/wp-content/uploads/2018/01/traffic-congestion.jpg"
    segment_image_from_url("https://www.msi.org.za/wp-content/uploads/2018/01/traffic-congestion.jpg")
