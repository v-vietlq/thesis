import torch
import torch.nn as nn
from models.backbone.resnet import *
import torchvision
from models.layers.avg_pool import FastAvgPool2d
from torch.autograd import Variable
import torch.nn.functional as F

backbone_filters = {
    # Resnet
    "resnet18": [64, 64, 128, 256, 512],
    "resnet34": [64, 64, 128, 256, 512],
    "resnet50": [64, 256, 512, 1024, 2048],
    "resnet101": [64, 256, 512, 1024, 2048],
    "resnet152": [64, 256, 512, 1024, 2048],

    # MobileNetV2
    "mobilenet_v2": [16, 24, 32, 96, 1280]
}


class EventNetwork(nn.Module):
    def __init__(self, encoder_name='resnet50', num_classes=23):
        super(EventNetwork, self).__init__()
        
        feature_extractor = getattr(
            torchvision.models, encoder_name)(pretrained=True)
        if encoder_name.startswith('resnet'):
            self.encoders = [
                nn.Sequential(feature_extractor.conv1,
                              feature_extractor.bn1, feature_extractor.relu),
                nn.Sequential(feature_extractor.maxpool,
                              feature_extractor.layer1),
                feature_extractor.layer2,
                feature_extractor.layer3,
                feature_extractor.layer4
            ]
        else:
            raise NotImplementedError('backbone should be resnet')

        self.encoders = nn.ModuleList(self.encoders)

        self.global_pool_layer = FastAvgPool2d(flatten=True)

        self.feat_in = backbone_filters[encoder_name]

        self.fc = nn.Linear(self.feat_in[-1], num_classes)

    def forward(self, x):
        features = []
        for module in self.encoders:
            x = module(x)
            features += [x]
        x0, x1, x2, x3, fea = features

        o = self.global_pool_layer(fea)

        o = self.fc(o)
        
        return o
    


class Aggregate(nn.Module):
    def __init__(self, sampled_frames=None, nvids=None, args=None):
        super(Aggregate, self).__init__()
        self.clip_length = sampled_frames
        self.nvids = nvids
        self.args = args


    def forward(self, x, filenames=None):
        nvids = x.shape[0] // self.clip_length
        x = x.view((-1, self.clip_length) + x.size()[1:])
        o = x.mean(dim=1)
        return o

class EventCnnLstm(nn.Module):
    def __init__(self, encoder_name='resnet101', num_classes =23,hidden_size=512):
        super(EventCnnLstm, self).__init__()
        feature_extractor = getattr(
            torchvision.models, encoder_name)(pretrained=True)
        if encoder_name.startswith('resnet'):
            self.encoders = [
                nn.Sequential(feature_extractor.conv1,
                              feature_extractor.bn1, feature_extractor.relu),
                nn.Sequential(feature_extractor.maxpool,
                              feature_extractor.layer1),
                feature_extractor.layer2,
                feature_extractor.layer3,
                feature_extractor.layer4
            ]
        else:
            raise NotImplementedError('backbone should be resnet')

        self.encoders = nn.ModuleList(self.encoders)

        self.global_pool_layer = FastAvgPool2d(flatten=True)

        self.feat_in = backbone_filters[encoder_name]

        
        self.rnn = nn.LSTM(
            input_size=self.feat_in[-1],
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True)
        
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        batch_size, time_steps, channels, height, width = x.size()
        c_in = x.view(batch_size * time_steps, channels, height, width)
        features = []
        for module in self.encoders:
            c_in = module(c_in)
            features += [c_in]
        x0, x1, x2, x3, fea = features

        o = self.global_pool_layer(fea)

        r_in = o.view(batch_size ,time_steps, -1)
        o, (_,_) = self.rnn(r_in)
        o = self.fc(o[:, -1, :])
        return o