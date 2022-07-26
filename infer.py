import torch
import argparse
import os
import matplotlib.pyplot as plt
import torchvision.utils
from PIL import Image
import numpy as np
from models.models import MTResnetAggregate


# ----------------------------------------------------------------------
# Parameters
parser = argparse.ArgumentParser(
    description='PETA: Photo album Event recognition using Transformers Attention.')
parser.add_argument('--model_path', type=str,
                    default='/content/drive2/checkpoints/event_importance_tresnet/version_2/checkpoints/best-epoch=119-mAP=67.21.ckpt')
parser.add_argument('--album_path', type=str,
                    default='../CUFED_split/images/test/0_11202221@N00')
parser.add_argument('--image_importance_pth', type=str,
                    default='../CUFED_split/image_importance.json')
parser.add_argument('--event_type_pth', type=str,
                    default='../CUFED_split/event_type.json')
parser.add_argument('---album_list', type=str,
                    default='filenames/test.txt')
parser.add_argument('--val_dir', type=str, default='./albums')
parser.add_argument('--num_classes', type=int, default=23)
parser.add_argument('--model_name', type=str, default='mtresnetaggregate')
parser.add_argument('--transformers_pos', type=int, default=0)
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--transform_type', type=str, default='squish')
parser.add_argument('--backbone', type=str, default='tresnet_m')
parser.add_argument('--album_sample', type=str, default='rand_permute')
parser.add_argument('--dataset_path', type=str, default='./data/ML_CUFED')
parser.add_argument('--dataset_type', type=str, default='ML_CUFED')
parser.add_argument('--path_output', type=str, default='./outputs')
parser.add_argument('--use_transformer', type=int, default=1)
parser.add_argument('--infer', type=int, default=1)
parser.add_argument('--album_clip_length', type=int, default=32)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--top_k', type=int, default=3)
parser.add_argument('--threshold', type=float, default=0.5)
parser.add_argument('--remove_model_jit', type=int, default=None)
parser.add_argument('--attention', type=str,
                    default='multihead')


def get_album(args):

    files = os.listdir(args.album_path)
    n_files = len(files)
    idx_fetch = np.linspace(0, n_files-1, args.album_clip_length, dtype=int)
    tensor_batch = torch.zeros(
        len(idx_fetch), args.input_size, args.input_size, 3)
    for i, id in enumerate(idx_fetch):
        im = Image.open(os.path.join(args.album_path, files[id]))
        im_resize = im.resize((args.input_size, args.input_size))
        np_img = np.array(im_resize, dtype=np.uint8)
        tensor_batch[i] = torch.from_numpy(np_img).float() / 255.0
    tensor_batch = tensor_batch.permute(0, 3, 1, 2).cuda()   # HWC to CHW
    # tensor_images = torch.unsqueeze(tensor_images, 0).cuda()
    montage = torchvision.utils.make_grid(tensor_batch).permute(1, 2, 0).cpu()
    return tensor_batch, montage


def inference(tensor_batch, model, classes_list, args):
    output, _ = model(tensor_batch)
    output = torch.squeeze(torch.sigmoid(output))
    np_output = output.cpu().detach().numpy()
    idx_sort = np.argsort(-np_output)
    # Top-k
    detected_classes = np.array(classes_list)[idx_sort][: args.top_k]
    scores = np_output[idx_sort][: args.top_k]
    # Threshold
    idx_th = scores > args.threshold
    return detected_classes[idx_th], scores[idx_th]


def display_image(im, tags, filename, path_dest):

    if not os.path.exists(path_dest):
        os.makedirs(path_dest)

    plt.figure()
    plt.imshow(im)
    plt.axis('off')
    plt.axis('tight')
    plt.rcParams["axes.titlesize"] = 16
    plt.title("Predicted classes: {}".format(tags))
    plt.savefig(os.path.join(path_dest, filename))


def load_model(net, path):
    if path is not None and path.endswith(".ckpt"):
        state_dict = torch.load(path, map_location='cpu')
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        compatible_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('net.'):
                compatible_state_dict[k.replace('net.', '')] = v
        net.load_state_dict(compatible_state_dict)

    return net


def main(classes_list):
    args = parser.parse_args()
    # net = EventNetwork(encoder_name='resnet101', num_classes=args.num_classes).cuda()
    # net.eval()
    net = MTResnetAggregate(args).cuda()
    net.eval()
    net = load_model(net, args.model_path)

    # Get album
    tensor_batch, montage = get_album(args)

    # Inference
    tags, confs = inference(tensor_batch, net, classes_list, args)

    # Visualization
    display_image(montage, tags, 'result.jpg', os.path.join(
        args.path_output, args.album_path.rsplit('/')[1]).replace("./albums", ""))

    # Actual validation process
    # print('loading album and doing inference...')
    # map = validate(model, val_loader, classes_list, args.threshold)
    # print("final validation map: {:.2f}".format(map))

    print('Done\n')


if __name__ == '__main__':
    classes_list = ['Architecture', 'BeachTrip', 'Birthday', 'BusinessActivity',
                    'CasualFamilyGather', 'Christmas', 'Cruise', 'Graduation', 'GroupActivity',
                    'Halloween', 'Museum', 'NatureTrip', 'PersonalArtActivity',
                    'PersonalMusicActivity', 'PersonalSports', 'Protest', 'ReligiousActivity',
                    'Show', 'Sports', 'ThemePark', 'UrbanTrip', 'Wedding', 'Zoo']
    main(classes_list)
