This is a package to deploy the yoloV5 pytorch object-detection model into an Azure FunctionsApp. 
the model will download itself from the internet. You can select the model size in yolo.py. increase size for more accuracy

The function requires the query param "img" which is a URL to an image to be segmented. It will return a list of vehicles identified

The package contains utilities to filter the Yolo model's output by:
* minimum_confidence ~ [0,1] default 0.2 - will ignore detections with a confidence value lower than this
* minimum_area_frac ~ [0,1] default 0.1 - the smallest fraction of area a vehicles bounding box can be. IE: return no vehicles which take up less than 10% of the image by area
* extra_classes ~ List(extra classes) default None - if provided, the model will return more than just the ["car", "bus", "truck"] classes it already  does.
* contains_point ~ List with (x,y) in normalised image coords (0 to 1). If provided, the funtion will only return vehicles who's bounding box contains the given point. This filter can be used to return only the vehicles which contains the center of a particular licence plate.

These filters are all optional, and can be omitted from the web request
The function should return the bounding box of all vehicles which pass the filters

An example usage for a local deployment (func start) would be:
http://localhost:7071/api/segment_image?img=https://img-ik.cars.co.za/carsimages/tr:n-stock_medf_200/7624452.jpg?v=138786372&contains_point=[0.5,0.5]&minimum_area_frac=0.4&min_confidence=0.35

Try changing the min_confidence to be higher, resulting in no vehicles returned

The Yolo model presented problems when converted to the ONNX format. While these problems were not insurmountable, 
to finish the MVP in time, I opted to just run the yolo model in pytorch. This increased the simplicity of the code, 
and stands as a template to deploy native pytorch models.


The downside of running the model in pytorch instead of ONNX is a decreased performance, and the need for pytorch and 
many other dependencies in the python environment. These dependencies can be seen in the requirements.txt


Azure FunctionsApp only supports CPU processing. As such, if additional performance is required, deplying this model to 
Azure Machine Learning service should speed up its inference. Deploying as an ONNX model will also 
significantly improve performance. So too will batching the inference