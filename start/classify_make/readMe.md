This package is for deployment of a vehicle recognition model which is in the ONNX file format - into an Azure functionApp
The onnx file is available on Azure in the vrmartifacts file share resource under model-checkpoints/
You must copy the onnx file into classify_make

The function requires the query param "img" which is a URL to an image to be segmented

Optionally, a crop parameter can be included in the web request. If provided, it should be a list of four items 
[center_x, center_y, width, height]. All of these values should be in normalised image coords, and they correspond to 
the outputs of the yolo model.

Thus, to plug the full pipeline together, the bounding box of the vehicle identified by the Yolo function - should be passed to this function

ONNX is a standarised format for deep neural networks which is highly optimised and runs on many platforms

A labels.json containing a list of vehicle labels should be provided. The order of these labels should match the 
label_ids which were used to train the recognition model

https://github.com/Tanzanite-MLOps/VehicleRecognition Contains the code to train a vehicle recognition model and corresponding ONNX file
