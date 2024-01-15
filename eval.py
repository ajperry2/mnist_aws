import sagemaker
from torchvision.datasets import MNIST
from torchvision import transforms
from sagemaker.pytorch import PyTorch
from sagemaker.pytorch.model import PyTorchModel

import gzip
import numpy as np
import random
import os
from glob import glob
from datetime import datetime


sagemaker_session = sagemaker.Session()
region = sagemaker_session.boto_region_name
bucket = sagemaker_session.default_bucket()
prefix = "sagemaker/DEMO-pytorch-mnist"
role = "SageMaker-DataSciNonFi"

# Get last model
model_jsons = glob("*_model.json")
model_jsons.sort(key = lambda date: datetime.strptime(date, "%m_%d_%Y_%H_%M_%S_model.json"))
last_model_json = model_jsons[-1]
with open(last_model_json, "r") as f:
    model_data = f.read().strip()

instance_type = "ml.c5.9xlarge"

image_uri = sagemaker.image_uris.retrieve(
    framework="pytorch",
    region=region,
    py_version="py39",
    image_scope="inference",
    version="1.13.1",
    instance_type=instance_type,
)

env_variables_dict = {
    "SAGEMAKER_TS_BATCH_SIZE": "1",
    "SAGEMAKER_TS_MAX_BATCH_DELAY": "10000000",
    "SAGEMAKER_TS_MIN_WORKERS": "1",
    "SAGEMAKER_TS_MAX_WORKERS": "1",
}

pytorch_model = PyTorchModel(
    model_data=model_data,
    role=role,
    image_uri=image_uri,
    source_dir="code",
    framework_version="1.13.1",
    entry_point="inference.py",
    env=env_variables_dict,
)


predictor = pytorch_model.deploy(
    initial_instance_count=1,
    instance_type=instance_type,
    serializer=sagemaker.serializers.JSONSerializer(),
    deserializer=sagemaker.deserializers.BytesDeserializer(),
)

data_dir = "data/MNIST/raw"
with gzip.open(os.path.join(data_dir, "t10k-images-idx3-ubyte.gz"), "rb") as f:
    images = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28).astype(np.float32)
mask = random.sample(range(len(images)), 16)  # randomly select some of the test images
mask = np.array(mask, dtype=int)
data = images[mask]
try:
    response = predictor.predict(np.expand_dims(data, axis=1))
    print("Raw prediction result:")
    print(response)
    print()
except Exception as e:
    print(e)
# 
list_response = json.loads(response.text)["PREDICTION"]
labeled_predictions = list(zip(range(10), response[0]))
# print("Labeled predictions: ")
# print(labeled_predictions)
# print()
# 
labeled_predictions.sort(key=lambda label_and_prob: 1.0 - label_and_prob[1])
print("Most likely answer: {}".format(labeled_predictions[0]))

sagemaker_session.delete_endpoint(endpoint_name=predictor.endpoint_name)

