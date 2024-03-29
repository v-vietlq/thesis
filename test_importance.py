import torch
import torch
import argparse
import os
import matplotlib.pyplot as plt
from torchmetrics import Precision
import torchvision.utils
from PIL import Image
import numpy as np
import timm
from models.clsnetwork import EventNetwork, EventCnnLstm
from models.siamesenetwork import SiameseNetwork
from models.models import MTResnetAggregate
import cv2
import json
import torchvision.transforms as T
from utils.ir_metrics import avg_precision_at_k, to_relevance_scores, precision_at_k


# ----------------------------------------------------------------------
# Parameters
parser = argparse.ArgumentParser(
    description='PETA: Photo album Event recognition using Transformers Attention.')
parser.add_argument('--model_path', type=str,
                    default='/content/drive2/checkpoints/event_importance_resnet50/version_3/checkpoints/best-epoch=69-mAP=65.87.ckpt')
parser.add_argument('--data_path', type=str,
                    default='../CUFED_split/images/test')
parser.add_argument('--image_importance_pth', type=str,
                    default='../CUFED_split/image_importance.json')
parser.add_argument('--event_type_pth', type=str,
                    default='../CUFED_split/event_type.json')
parser.add_argument('--album_list', type=str,
                    default='filenames/test.txt')
parser.add_argument('--album_name', type=str,
                    default='1_36030443@N06')
parser.add_argument('--val_dir', type=str, default='./albums')
parser.add_argument('--num_classes', type=int, default=23)
parser.add_argument('--model_name', type=str, default='mtresnetaggregate')
parser.add_argument('--transformers_pos', type=int, default=0)
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--transform_type', type=str, default='squish')
parser.add_argument('--backbone', type=str, default='resnet50')
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
parser.add_argument('--t', type=int, default=5)
parser.add_argument('--threshold', type=float, default=0.5)
parser.add_argument('--remove_model_jit', type=int, default=None)
parser.add_argument('--attention', type=str,
                    default='multihead')

test_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ),
])


def get_album(args, album_name, t):
    files = os.listdir(os.path.join(args.data_path, album_name))

    cls_dict = json.load(open(args.event_type_pth))
    event = cls_dict[album_name]

    scores_dict = json.load(open(args.image_importance_pth))

    scores = {img[1]: img[2]
              for imgs in scores_dict.values() for img in imgs}

    images = [k.split('/')[1] for k, v in sorted(scores.items(),
                                                 key=lambda item: item[1]) if album_name in k]

    top_t_labels = images[-(int(len(files)*t / 100) + 1):]

    n_files = len(files)
    tensor_batch = torch.zeros(
        n_files, args.input_size, args.input_size, 3)
    for i in range(n_files):
        im = Image.open(os.path.join(
            args.data_path, album_name, files[i])).convert('RGB')
        im_resize = im.resize((args.input_size, args.input_size))
        np_img = np.array(im_resize, dtype=np.uint8)
        tensor_batch[i] = torch.from_numpy(np_img).float() / 255.0

    files = [file.replace('.jpg', '') for file in files]
    tensor_batch = tensor_batch.permute(0, 3, 1, 2).cuda()   # HWC to CHW

    return tensor_batch, np.array(top_t_labels), np.array(files), event


def plot_image(i, predictions_array, true_label, img, class_names):
    img = img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)

    plt.xlabel("{} {:2.0f}% ({threshold})".format(class_names[predicted_label],
                                                  100 *
                                                  np.max(predictions_array),
                                                  true_label))


def inference(tensor_batch, model, classes_list, args):

    out = model(tensor_batch)
    print(out.shape)
    output = torch.squeeze(torch.sigmoid(model(tensor_batch)))
    np_output = output.cpu().detach().numpy()
    idx_sort = np.argsort(-np_output)
    # Top-k
    detected_classes = np.array(classes_list)[idx_sort][: args.top_k]
    scores = np_output[idx_sort][: args.top_k]

    # Threshold
    idx_th = scores > args.threshold
    print(detected_classes[idx_th], scores[idx_th])
    return detected_classes[idx_th], scores[idx_th]


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

    # mAP = MAP(net, args.album_list, 30, args)
    # mp = mP(net, args.album_list, 30, args)
    # print('----')
    # print("mAP: ", mAP)
    # print("mP: ", mp)
    # print('----')
    montage_target, montage_pred, tags, event = get_result(
        net, args.album_name, args.t, args, classes_list)
    display_image(montage_target, montage_pred, tags, event)


def P_Per_Album(net, album_name, t, args):
    tensor_batch, target, files, event = get_album(args, album_name, t)
    with torch.no_grad():
        _, output = net(tensor_batch)
        output = torch.squeeze(torch.sigmoid(output))

    np_output = output.cpu().detach().numpy()

    idx_sort = np.argsort(-np_output)

    idx_th = idx_sort[:int(len(files)*t / 100) + 1]

    pred = files[idx_th]
    return precision_at_k(target, pred)


def AP_Per_Album(net, album_name, t, args):
    tensor_batch, target, files, event = get_album(args, album_name, t)
    with torch.no_grad():
        _, output = net(tensor_batch)
        output = torch.squeeze(torch.sigmoid(output))

    np_output = output.cpu().detach().numpy()

    idx_sort = np.argsort(-np_output)

    idx_th = idx_sort[:int(len(files)*t / 100) + 1]

    pred = files[idx_th]

    return avg_precision_at_k(target, pred)


def mP(net, album_list, t, args):
    result = []
    albums = np.loadtxt(album_list, dtype='str', delimiter='\n')
    for album in albums:
        p_album = P_Per_Album(net, album, t, args)
        result.append(p_album)
    return np.mean(result)


def MAP(net, album_list, t, args):
    result = []
    albums = np.loadtxt(album_list, dtype='str', delimiter='\n')
    for album in albums:
        ap_album = AP_Per_Album(net, album, t, args)
        result.append(ap_album)
    return np.mean(result)


def display_image(im1, im2, tags, event):
    rows = 2
    columns = 2
    fig = plt.figure(figsize=(15, 7))
    plt.axis('off')
    plt.title('event class')
    plt.rcParams["axes.titlesize"] = 16
    plt.title("Predicted classes: {} \n True classes: {}".format(tags, event))
    # Adds a subplot at the 1st position

    fig.add_subplot(2, 2, 1)
    # showing image
    plt.imshow(im1)
    plt.axis('off')
    plt.title("target")
    # Adds a subplot at the 2nd position
    fig.add_subplot(2, 2, 2)
    # showing image
    plt.imshow(im2)
    plt.axis('off')
    plt.title("pred")

    fig.savefig('result.png')


def get_result(net, album_name, t, args, classes_list):
    tensor_batch, target, files, event = get_album(args, album_name, t)

    with torch.no_grad():
        event_output, output = net(tensor_batch)
        output = torch.squeeze(torch.sigmoid(output))
        event_output = torch.squeeze(torch.sigmoid(event_output))

    np_eventoutput = event_output.cpu().detach().numpy()
    idx_eventsort = np.argsort(-np_eventoutput)
    # Top-k
    detected_classes = np.array(classes_list)[idx_eventsort][: args.top_k]
    scores_event = np_eventoutput[idx_eventsort][: args.top_k]
    # Threshold
    idx_event = scores_event > args.threshold
    # detected_classes[idx_th], scores[idx_th]

    np_output = output.cpu().detach().numpy()

    idx_sort = np.argsort(-np_output)

    idx_th = idx_sort[:int(len(files)*t / 100) + 1]

    target_batch = torch.zeros(
        len(target), args.input_size, args.input_size, 3)

    for i in range(len(target)):
        im = Image.open(os.path.join(
            args.data_path, album_name, target[i] + '.jpg')).convert('RGB')
        im_resize = im.resize((args.input_size, args.input_size))
        np_img = np.array(im_resize, dtype=np.uint8)
        target_batch[i] = torch.from_numpy(np_img).float() / 255.0
    target_batch = target_batch.permute(0, 3, 1, 2).cuda()   # HWC to CHW
    # tensor_images = torch.unsqueeze(tensor_images, 0).cuda()
    montage_target = torchvision.utils.make_grid(
        target_batch).permute(1, 2, 0).cpu()

    pred = files[idx_th]
    pred_batch = torch.zeros(
        len(pred), args.input_size, args.input_size, 3)

    for i in range(len(pred)):
        im = Image.open(os.path.join(
            args.data_path, album_name, pred[i] + '.jpg')).convert('RGB')
        im_resize = im.resize((args.input_size, args.input_size))
        np_img = np.array(im_resize, dtype=np.uint8)
        pred_batch[i] = torch.from_numpy(np_img).float() / 255.0

    pred_batch = pred_batch.permute(0, 3, 1, 2).cuda()   # HWC to CHW
    # tensor_images = torch.unsqueeze(tensor_images, 0).cuda()
    montage_pred = torchvision.utils.make_grid(
        pred_batch).permute(1, 2, 0).cpu()

    return montage_target, montage_pred, detected_classes[idx_event], event


if __name__ == '__main__':
    classes_list = ['Architecture', 'BeachTrip', 'Birthday', 'BusinessActivity',
                    'CasualFamilyGather', 'Christmas', 'Cruise', 'Graduation', 'GroupActivity',
                    'Halloween', 'Museum', 'NatureTrip', 'PersonalArtActivity',
                    'PersonalMusicActivity', 'PersonalSports', 'Protest', 'ReligiousActivity',
                    'Show', 'Sports', 'ThemePark', 'UrbanTrip', 'Wedding', 'Zoo']
    main(classes_list)
