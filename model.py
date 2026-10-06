import timm
import torch
import torch.nn as nn


def list_models():
    model = timm.list_models("resnet50*", pretrained=True)

    print(model)


def build_model(model_name, num_classes):
    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    return model

if __name__ == "__main__":
    # list_models()
    model = build_model("resnet50.tv_in1k", num_classes=3)  # imagenet 1000 = 1K
    # print(model)

    dummy_input = torch.randn(
        16, 3, 224, 224
    )  # [Batch, channel, height, width]
    output = model(dummy_input)
    print(output.shape)  # [Batch, num_classes]
    print(output)

    # input = [Batch, channel, height, width]
    # output = [Batch, num_classes]