from models.clsnetwork import EventCnnLstm
from models.siamesenetwork import SiameseNetwork
import torch
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
    # net = FineTuneLstmModel(arch='resnet101',num_classes = 23,lstm_layers=1, hidden_size= 512,fc_size=512).cuda()
    net1 = EventCnnLstm(encoder_name='resnet101', num_classes=23, hidden_size=512).cuda()
    # net = SiameseNetwork(backbone='alexnet')
    x = torch.rand(4, 32 ,3,224,224).cuda()
    o = net1(x)
    print(o.shape)
    # print(o[0].shape)