import json
import os
from mnist import Net
import torch

JSON_CONTENT_TYPE = "application/json"


def model_fn(model_dir):
    print("LOADING MODEL")
    try:
        model = Net()
        model.load_state_dict(torch.load(model_dir))
        model.eval()
    except Exception as e:
        print(e)
    return (model,)


def input_fn(serialized_input_data, content_type=JSON_CONTENT_TYPE):
    print("INPUT1")
    if content_type == JSON_CONTENT_TYPE:
        print("INPUT2")
        input_data = json.loads(serialized_input_data)
        return input_data

    else:
        raise Exception("Requested unsupported ContentType in Accept: " + content_type)
        return


def predict_fn(input_data, model_pack):

    print("Got input Data: {}".format(input_data))
    model = model_pack[0]


    inputs = torch.tensor(input_data)
    with torch.no_grad():
        output = model(inputs)

    print("PRED", output)


    return {"PREDICTION":output.detach().numpy().tolist()}


def output_fn(prediction_output, accept=JSON_CONTENT_TYPE):
    print("PREDICTION", prediction_output)

    if accept == JSON_CONTENT_TYPE:
        return json.dumps(prediction_output), accept

    raise Exception("Requested unsupported ContentType in Accept: " + accept)