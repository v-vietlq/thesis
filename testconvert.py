import torch
import caffemodel2pytorch

model = caffemodel2pytorch.Net(
    prototxt='/home/vietlq4/Downloads/CUFED_split/python_deploy.prototxt',
    weights='/home/vietlq4/Downloads/CUFED_split/resnet_model.caffemodel',
    caffe_proto='https://raw.githubusercontent.com/BVLC/caffe/master/src/caffe/proto/caffe.proto'
)
model.cuda()
model.eval()
print(model)
torch.set_grad_enabled(False)
