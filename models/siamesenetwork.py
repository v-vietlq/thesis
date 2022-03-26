import torch
import torch.nn as nn
from torchvision import models


class SiameseNetwork(nn.Module):
    def __init__(self, backbone='resnet101'):
        super(SiameseNetwork, self).__init__()
        self.backbone = models.resnet101(pretrained=True)
        modules = list(self.backbone.children())[:-1]
        self.features = nn.Sequential(*modules)
        self.fc1 = nn.Sequential(
            nn.Linear(2048, 500),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),
            nn.Linear(500, 1),
        )

    def forward_once(self, x):
        x = self.features(x)
        x = x.reshape(x.size(0), -1)
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
