import argparse
import os
from util import util
import torch
import models
import data


class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument('--scaled_width', type=int, default=1248, help='input width of the model')
        parser.add_argument('--scaled_height', type=int, default=384, help='input height of the model')
        parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment')
        parser.add_argument('--dataset', type=str, default='roadlanemark', help='chooses which dataset to load.')
        parser.add_argument('--num_labels', type=int, help='number of labels')
        parser.add_argument('--model', type=str, default='unet_resnet34', help='choose the backbone for semantic segmentation')
        parser.add_argument('--pretrained', type=str, help='pretrained model path (.ckpt or .pth)')
        parser.add_argument('--num_threads', default=8, type=int, help='# threads for loading data')
        parser.add_argument('--seed', type=int, default=0, help='seed for random generators')
        parser.add_argument('--use_ocr', action='store_true', help='apply OCR')

        parser.add_argument('--normalization', default='imagenet', help='normalization type: imagenet, default (1/255)')
        parser.add_argument('--val_root', required=True, help='root folder containing images for validation')
        parser.add_argument('--val_list', required=True, help='.txt file containing validation image list')
        parser.add_argument('--val_data_sign', required=True, nargs="+", type=str, help='list of data_type. e.g vistas bdd')

        parser.add_argument('--save_dir', type=str, default='./checkpoints', help='where checkpoints and log are save. The final saved dir would be: <save_dir>/<name>/version_<0,1,2...>/')
        parser.add_argument('--batch_size', type=int, default=2, help='input batch size')

        # Distributed
        parser.add_argument('--gpus', type=str, default='0', help='gpu ids for training, testing. e.g. 0 or 0,1,2')
        parser.add_argument('--accelerator', type=str, default='ddp', help='DataParallel (dp), DistributedDataParallel (ddp)')
        self.initialized = True
        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        self.parser = parser

        return parser.parse_args()

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        # expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        # util.mkdirs(expr_dir)
        # file_name = os.path.join(expr_dir, 'opt.txt')
        # with open(file_name, 'wt') as opt_file:
        #     opt_file.write(message)
        #     opt_file.write('\n')

    def parse(self):
        opt = self.gather_options()

        self.print_options(opt)

        # set gpu ids
        # str_ids = opt.gpu_ids.split(',')
        # opt.gpu_ids = []
        # for str_id in str_ids:
        #     id = int(str_id)
        #     if id >= 0:
        #         opt.gpu_ids.append(id)
        # if len(opt.gpu_ids) > 0:
        #     torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt