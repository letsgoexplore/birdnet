import torch
import torch.nn as nn

class resnet50(nn.Module):
    def __init__(self, num_classes):
        super(resnet50, self).__init__()
        self.resnet = torch.hub.load('pytorch/vision:v0.9.0', 'resnet50', pretrained=True)
        self.fc = nn.Linear(1000, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        x = self.fc(x)
        return x