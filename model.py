import timm
import torch
import torch.nn as nn


def list_models():
    model = timm.list_models("resnet50*", pretrained=True)

    print(model)


def build_model(model_name, num_classes, pretrained=True):
    # 允許離線檢查時停用預訓練權重；正式訓練預設不變。
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    return model


# 單張 224×224 輸入的 Conv2d/Linear MACs；FLOPs 約為 2×MACs。
# 不包含 BN、activation、pooling、bias 與 residual add，故為估算而非完整運算量。
@torch.no_grad()
def measure_model_resources(model):
    macs = 0
    handles = []
    training = model.training

    def count_ops(module, inputs, output):
        nonlocal macs
        if isinstance(module, nn.Conv2d):
            kernel_ops = (module.in_channels // module.groups) * module.kernel_size[0] * module.kernel_size[1]
            macs += output.numel() * kernel_ops
        elif isinstance(module, nn.Linear):
            macs += output.numel() * module.in_features

    try:
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                handles.append(module.register_forward_hook(count_ops))
        model.eval()
        parameter = next(model.parameters())
        model(torch.zeros(1, 3, 224, 224, device=parameter.device, dtype=parameter.dtype))
    finally:
        for handle in handles:
            handle.remove()
        model.train(training)
    return {
        "parameters": sum(p.numel() for p in model.parameters()),
        "macs_conv_linear": macs,
        "flops_conv_linear_estimate": 2 * macs,
    }

if __name__ == "__main__":
    # list_models()
    # 此 AOI 任務有六類，示範輸出也統一為六類。
    model = build_model("resnet50.tv_in1k", num_classes=6)  # imagenet 1000 = 1K
    # print(model)

    dummy_input = torch.randn(
        16, 3, 224, 224
    )  # [Batch, channel, height, width]
    output = model(dummy_input)
    print(output.shape)  # [Batch, num_classes]
    print(output)

    # input = [Batch, channel, height, width]
    # output = [Batch, num_classes]
