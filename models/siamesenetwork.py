import torch
import torch.nn as nn
import timm
from .layers.avg_pool import FastAvgPool2d, FastAdaptiveAvgPool2d


class SiameseNetwork(nn.Module):
    def __init__(self, encoder_name='resnet101'):
        super(SiameseNetwork, self).__init__()
        self.feature_extraction = timm.create_model(
            model_name=encoder_name, pretrained=True)
        self.global_pool = FastAdaptiveAvgPool2d(flatten=True)
        self.fc1 = nn.Sequential(
            nn.Linear(self.feature_extraction.num_features, 500),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),
            nn.Linear(500, 1),
        )

    def forward(self, x):
        x = self.feature_extraction.forward_features(x)
        x = self.global_pool(x)
        x = self.fc1(x)
        return x


if __name__ == '__main__':
    x = torch.rand(16, 3, 224, 224)
    m = SiameseNetwork()
    o = m(x)
    print(o.shape)
