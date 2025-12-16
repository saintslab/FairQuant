from typing import Tuple
import torch
import torch.nn as nn
import torchvision.models as tvm

# Try to import timm
try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False


def get_model(name: str, num_classes: int, pretrained: bool = True, image_size: int = 224) -> nn.Module:
    name = name.lower()
    
    if name == "resnet18":
        weights = tvm.ResNet18_Weights.DEFAULT if pretrained else None
        m = tvm.resnet18(weights=weights)
        in_features = m.fc.in_features
        m.fc = nn.Linear(in_features, num_classes)
        return m
        
    elif name == "resnet34":
        weights = tvm.ResNet34_Weights.DEFAULT if pretrained else None
        m = tvm.resnet34(weights=weights)
        in_features = m.fc.in_features
        m.fc = nn.Linear(in_features, num_classes)
        return m
        
    elif name == "resnet50":
        weights = tvm.ResNet50_Weights.DEFAULT if pretrained else None
        m = tvm.resnet50(weights=weights)
        in_features = m.fc.in_features
        m.fc = nn.Linear(in_features, num_classes)
        return m

    elif HAS_TIMM:
        try:
            # timm automatically handles the head replacement via num_classes
            m = timm.create_model(name, pretrained=pretrained, num_classes=num_classes)
            return m
        except Exception as e:
            raise ValueError(f"Could not load model '{name}' from torchvision or timm. Error: {e}")
    
    else:
        raise ValueError(f"Unknown model '{name}'. Supported models: resnet18, resnet34, resnet50, vgg16, vgg19.")