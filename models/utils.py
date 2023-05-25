import torch


def weights_init(model: torch.nn.Module):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(model.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(model.weight, 1.0, 0.02)
        torch.nn.init.zeros_(model.bias)
