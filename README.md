Article: https://medium.com/pytorch/efficient-serverless-deployment-of-pytorch-models-on-azure-dc9c2b6bfee7


# Deploy ONNX Models to Azure Functions

Samples for serverless deployment of ONNX models to Azure Functions. 

## Pre-requisites

Before you begin, you must have the following:

1. An Azure account with an active subscription. [Create an account for free](https://azure.microsoft.com/free).

On a Linux system or Windows (WSL or WSL2) ensure you have the following installed:

2. The [Azure Functions Core Tools](https://docs.microsoft.com/en-us/azure/azure-functions/functions-run-local#v2)
3. The [Azure CLI](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli) 
4. Python >3.7

Note: You can also use Azure Cloud Shell which comes preinstalled with Azure CLI and Functions Core Tools. 

## Develop and Test Locally

You can develop for serverless deployment on Azure functions on your Linux machine locally or a development VM (like the Azure Data Science VM) on the cloud or Azure Cloud Shell. Run the following commands to setup your Azure Functions project locally.

1. Copy the deployment code 

git clone https://github.com/Tanzanite-MLOps/functions-deploy-pytorch-onnx.git ~/functions-deploy-pytorch-onnx

2. cd start

3. Initialize Function App
func init --worker-runtime python


4. Create and activate Python virtualenv to setup ONNX runtime along with dependencies

python -m venv .venv
source .venv/bin/activate
pip install --no-cache-dir -r requirements.txt  

* Alternatively you can use conda


5. Copy your ONNX model file (which should have a name model.onnx)  built from training the Pytorch model  and converting to ONNX into  the "start/classify_make" directory within your Function App project
    The code to create such a model can be found: https://github.com/Tanzanite-MLOps/VehicleRecognition
6. Run the test locally:

```
func start
```
In a browser on your machine you can test the local Azure Function by visiting: 

"http://localhost:7071/api/classify_make?img=http://3.bp.blogspot.com/-S1scRCkI3vY/UHzV2kucsPI/AAAAAAAAA-k/YQ5UzHEm9Ss/s1600/Grizzly%2BBear%2BWildlife188.jpg"


## Create Resources in Azure and Publish

1. Create Azure Function App using Azure CLI
You can also just use the existing vehicleRecognitionFunc and skip this step

```
# If you have not logged into Azure CLI. you must first run "az login" and follow instructions
az group create --name vehicleRecognitionFunc --location eastus
az storage account create --name inferencestorage -l eastus --sku Standard_LRS -g vehicleRecognitionFunc
az functionapp create --name vehicleRecognitionFunc -g vehicleRecognitionFunc --consumption-plan-location eastus --storage-account inferencestorage --runtime python --runtime-version 3.9 --functions-version 3 --disable-app-insights --os-type Linux
```
2. Publish to Azure

```
# Install a local copy of ONNX runtime and dependencies to push to Azure Functions Runtime
pip install  --target="./.python_packages/lib/site-packages"  -r requirements.txt

# Publish Azure function to the 
func azure functionapp publish vehicleRecognitionFunc --no-build
```
Repeating the 'func azure functionapp publish' call will refresh the online function and sync it to your local code.
Renaming the packages will rename the functions automatically.
Various function args, such as trigger type - can be found in the packages/function.json

It will take a few minutes to publish and bring up the Azure functions with your ONNX model deployed and exposed as a http endpoint.  Then you can find the URL by running the following command:  ```func azure functionapp list-functions vehicleRecognitionFunc --show-keys``` . Append ```&img=[[Your Image URL to run thru model]]``` to the URL on a browser to get predictions from the model running in the Azure Functions. 
func azure functionapp list-functions vehicleRecognitionFunc --show-keys



## Deleting Resources
To delete all the resources (and avoiding any charges) after you are done, run the following Azure CLI command:
```
az group delete --name vehicleRecognitionFunc --yes

```
Azure Functions provides other options like auto scaling, larger instances, monitoring with Application Insights etc. 

## Notes

1. If you are using Windows WSL or WSL2 and face authentication issues while deploying Function App to Azure, one of the main reason is your clock in WSL (Linux) may be out of sync with the underlying Windows host. You can synch it by running ```sudo hwclock -s``` in WSL. 

