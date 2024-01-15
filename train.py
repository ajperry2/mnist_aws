import sagemaker
from torchvision.datasets import MNIST
from torchvision import transforms
from sagemaker.pytorch import PyTorch
from datetime import datetime
import json

sagemaker_session = sagemaker.Session()
region = sagemaker_session.boto_region_name
bucket = sagemaker_session.default_bucket()
prefix = "sagemaker/DEMO-pytorch-mnist"
role = "SageMaker-DataSciNonFi"

MNIST.mirrors = [
    f"https://sagemaker-example-files-prod-{region}.s3.amazonaws.com/datasets/image/MNIST/"
]
MNIST(
    "data",
    download=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    ),
)
inputs = sagemaker_session.upload_data(path="data", bucket=bucket, key_prefix=prefix)
print("input spec (in this case, just an S3 path): {}".format(inputs))

estimator = PyTorch(
    entry_point="mnist.py",
    role=role,
    py_version="py38",
    framework_version="1.11.0",
    source_dir="code",
    instance_count=2,
    instance_type="ml.c5.2xlarge",
    hyperparameters={"epochs": 1, "backend": "gloo"},
)

estimator.fit({"training": inputs})

now = datetime.now() # current date and time
date_time = now.strftime("%m_%d_%Y_%H_%M_%S")

with open(f"{date_time}_model.json", "w+") as f:
    f.write(estimator.model_data)
