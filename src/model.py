import torch
import torch.nn as nn

from torchvision import models, transforms

def add_swin_channels(model, channels, dim=96):
    weight_indices = {'red': 0, 'green': 1, 'blue': 2, 'nir': 0, 'red_edge': 0}
    weight = model.features[0][0].weight.clone()
    model.features[0][0] = nn.Conv2d(len(channels), dim, kernel_size=(4, 4), stride=(4, 4))
    with torch.no_grad():
        for i, channel in enumerate(channels):
            model.features[0][0].weight[:, i] = weight[:, weight_indices[channel]]
    return model

def add_efficient_channels(model, channels, dim=48):
    weight_indices = {'red': 0, 'green': 1, 'blue': 2, 'nir': 0, 'red_edge': 0}
    weight = model.features[0][0].weight.clone()
    model.features[0][0] = nn.Conv2d(5, 48, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    with torch.no_grad():
        for i, channel in enumerate(channels):
            model.features[0][0].weight[:, i] = weight[:, weight_indices[channel]]
    return model


def add_resnet_channels(model, channels):
    weight_indices = {'red': 0, 'green': 1, 'blue': 2, 'nir': 0, 'red_edge': 0}
    weight = model.conv1.weight.clone()
    model.conv1 = nn.Conv2d(len(channels), 64, kernel_size=7, stride=2, padding=3, bias=False)
    with torch.no_grad():
        for i, channel in enumerate(channels):
            model.conv1.weight[:, i] = weight[:, weight_indices[channel]]
    return model

def full_precision_model(channels,model_name='resnet18'):
    if model_name == 'resnet18':
        model = models.resnet18(weights='ResNet18_Weights.IMAGENET1K_V1')
        model = add_resnet_channels(model, channels)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
    elif model_name == 'resnet50':
        model = models.resnet50(weights='ResNet50_Weights.IMAGENET1K_V1')
        model = add_resnet_channels(model, channels)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
    elif model_name == 'swin_t':
        model = models.swin_t(weights='Swin_T_Weights.IMAGENET1K_V1')
        model = add_swin_channels(model, channels)
        model.head = nn.Linear(in_features=768, out_features=2, bias=True)
    elif model_name == 'swin_s':
        model = models.swin_s(weights='Swin_S_Weights.IMAGENET1K_V1')
        model = add_swin_channels(model, channels)
    elif model_name == 'swin_b':
        model = models.swin_b(weights='Swin_B_Weights.IMAGENET1K_V1')
        model = add_swin_channels(model, channels, dim=128)
        model.head = nn.Linear(in_features=1024, out_features=2, bias=True)
    elif model_name == 'efficientnet_b4':
        model = models.efficientnet_b4(weights='EfficientNet_B4_Weights.IMAGENET1K_V1')
        model = add_efficient_channels(model, channels)
        model.classifier = nn.Linear(in_features=1792, out_features=2, bias=True)
    return model