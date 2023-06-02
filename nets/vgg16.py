import torch
import torch.nn as nn

class vgg19(nn.Module):
    def __init__(self, num_classes):
        super(vgg19, self).__init__()
        self.vgg = torch.hub.load('pytorch/vision:v0.9.0', 'vgg19', pretrained=True)
        self.fc = nn.Linear(1000, num_classes)

    def forward(self, x):
        x = self.vgg(x)
        x = self.fc(x)
        return x