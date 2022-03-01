import torch
import torch.nn as nn
from torch.nn.modules.activation import Sigmoid
import torchvision
from models.backbone.alexnet import alexnet


class SiameseNetwork(nn.Module):
    def __init__(self, backbone='alexnet'):
        super(SiameseNetwork, self).__init__()
        self.backbone = alexnet(pretrained=True)
        self.features = self.backbone.features
        self.avgpool = self.backbone.avgpool
        self.classifier = self.backbone.classifier[:-2]
        self.fc1 = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 23),
        )

    def forward_once(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x = self.fc1(x)
        return x

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


if __name__ == '__main__':
    x = torch.rand(16, 3, 224, 224)
    y = torch.rand(16, 3, 224, 224)
    m = SiameseNetwork()
    o = m.forward_once(x)
    print(o.shape)
