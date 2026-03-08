import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights


def get_model(num_classes=2):

    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model