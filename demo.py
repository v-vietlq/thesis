from models.clsnetwork import EventCnnLstm
from models.siamesenetwork import SiameseNetwork
from models.models import MTResnetAggregate, TResnetMLDecoder
import torch
import numpy as np
from options.train_options import TrainOptions
from utils.loss import *
# if __name__ == '__main__':
#     x1 = torch.randn(16, 3, 224, 224).cuda()
#     y1 = torch.randn(16, 1).cuda()
#     x2 = torch.randn(16, 3, 224, 224).cuda()
#     y2 = torch.randn(16, 1).cuda()
#     m = SiameseNetwork().cuda()
#     o1, o2 = m(x1, x2)
#     l = PiecewiseLoss().cuda()
#     print(o1.shape, o2.shape, y1.shape, y2.shape)
#     loss = l(o1,o2,y1, y2)
#     print(loss)
if __name__ == '__main__':
    # train_opt = TrainOptions().parse()
    # np.random.seed(train_opt.seed)
    # torch.manual_seed(train_opt.seed)
    # torch.cuda.manual_seed(train_opt.seed)
    # train_opt.phase = 'train'

    # val_opt = TrainOptions().parse()
    # val_opt.phase = 'val'
    # val_opt.batch_size = 1
    # # net = FineTuneLstmModel(arch='resnet101',num_classes = 23,lstm_layers=1, hidden_size= 512,fc_size=512).cuda()
    # # net1 = EventCnnLstm(encoder_name='resnet101', num_classes=23, hidden_size=512).cuda()
    # net = MTResnetAggregate(args=train_opt)
    net = TResnetMLDecoder()
    # net = SiameseNetwork(backbone='alexnet')
    x = torch.rand(32, 3, 224, 224)

    o = net(x)
    print(o.shape)
    # print(o[0].shape)
