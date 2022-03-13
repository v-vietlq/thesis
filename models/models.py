import imp
import torch.nn as nn
from models.tresnet.tresnet import TResNet
# from models.utils.registry import register_model
from .layers.avg_pool import FastAvgPool2d
from models.aggregate.layers.frame_pooling_layer import Aggregate
from models.aggregate.layers.transformer_aggregate import TAggregate
import timm
# from src.models.resnet.resnet import Bottleneck as ResnetBottleneck
# from models.resnet.resnet import ResNet

# __all__ = ['MTResnetAggregate']


class fTResNet(nn.Module):

    def __init__(self,encoder_name ='tresnet_m', num_classes=23, aggregate=None, *args, **kwargs):
        super(fTResNet, self).__init__(*args, **kwargs)
        self.aggregate = aggregate
        self.feature_extraction =  timm.create_model(model_name=encoder_name, pretrained=True)
        self.head = nn.Linear(self.feature_extraction.num_features, num_classes)
        self.global_pool = FastAvgPool2d(flatten=True)
    def forward(self, x, filenames=None):
        x = self.feature_extraction.forward_features(x)
        # x = self.body(x)
        self.embeddings = self.global_pool(x)
        if self.aggregate:
            if isinstance(self.aggregate, TAggregate):
                self.embeddings, self.attention = self.aggregate(
                    self.embeddings, filenames)
                logits = self.head(self.embeddings)
            else:  # CNN aggregation:
                logits = self.head(self.embeddings)
                logits = self.aggregate(nn.functional.softmax(logits, dim=1))

        # if attn_mat is None:
        #     return logits
        # else:
        #     return (logits, attn_mat)
        return logits


# class fResNet(ResNet):
#     def __init__(self, aggregate=None, **kwargs):
#         super().__init__(**kwargs)
#         self.aggregate = aggregate

#     def forward(self, x):
#         x = self.body(x)
#         if self.aggregate:
#             x = self.head.global_pool(x)
#             x, attn_weight = self.aggregate(x)
#             logits = self.head.fc(self.head.FlattenDropout(x))

#         else:
#             logits = self.head(x)
#         return logits


def MTResnetAggregate(args):
    in_chans = 3
    aggregate = None
    if args.use_transformer:
        aggregate = TAggregate(args.album_clip_length, args=args)
    else:
        aggregate = Aggregate(args.album_clip_length, args=args)

    model = fTResNet(encoder_name ='tresnet_m', num_classes=23,aggregate=aggregate)

    return model

