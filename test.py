import torch
import torch
import argparse
import os
import matplotlib.pyplot as plt
import torchvision.utils
from PIL import Image
import numpy as np
# import timm
from models.clsnetwork import EventNetwork, EventCnnLstm
import cv2
import json
import torchvision.transforms as T
from utils.utils import *
from dataset import *
from models.models import MTResnetAggregate
from options.train_options import TrainOptions

# ----------------------------------------------------------------------
# Parameters
parser = argparse.ArgumentParser(
    description='PETA: Photo album Event recognition using Transformers Attention.')
parser.add_argument('--model_path', type=str,
                    default='/content/drive2/checkpoints/event_transformer/version_42/checkpoints/best-epoch=02-mAP=51.20.ckpt')
parser.add_argument('--album_path', type=str,
                    default='/home/ubuntu/datasets/CUFED/images/15_74648938@N00')
# /Graduation') # /0_92024390@N00')
parser.add_argument('---album_list', type=str,
                        default='filenames/test.txt')
parser.add_argument('--event_type_pth', type=str,
                            default='../CUFED/event_type.json')
parser.add_argument('--val_dir', type=str, default='./albums')
parser.add_argument('--num_classes', type=int, default=23)
parser.add_argument('--model_name', type=str, default='mtresnetaggregate')
parser.add_argument('--transformers_pos', type=int, default=1)
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--transform_type', type=str, default='squish')
parser.add_argument('--album_sample', type=str, default='rand_permute')
parser.add_argument('--dataset_path', type=str, default='./data/ML_CUFED')
parser.add_argument('--dataset_type', type=str, default='ML_CUFED')
parser.add_argument('--path_output', type=str, default='./outputs')
parser.add_argument('--use_transformer', type=int, default=1)
parser.add_argument('--album_clip_length', type=int, default=32)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--top_k', type=int, default=3)
parser.add_argument('--threshold', type=float, default=0.85)
parser.add_argument('--remove_model_jit', type=int, default=None)


test_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
    ])

def get_album(args):

    files = os.listdir(args.album_path)
    album_name = args.album_path.rsplit('/', 1)[1]
    cls_dict = json.load(open(args.event_type_pth))
    event = cls_dict[album_name]
    n_files = len(files)
    print(n_files)
    frames = []
    idx_fetch = np.linspace(0, n_files-1, 32, dtype=int)
    for i, id in enumerate(idx_fetch):
        im = Image.open(os.path.join(args.album_path,files[id])).convert('RGB')
        im = test_transform(im)
        frames.append(im)
    
    # tensor_batch = tensor_batch.permute(0, 1, 2, 3)   # HWC to CHW
    tensor_batch = torch.unsqueeze(torch.stack(frames), 0).cuda()
    # montage = torchvision.utils.make_grid(tensor_batch).permute(1, 2, 0).cpu()
    return tensor_batch, event

def plot_image(i, predictions_array, true_label, img, class_names):
    img =  img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)

    plt.xlabel("{} {:2.0f}% ({threshold})".format(class_names[predicted_label],
                                    100*np.max(predictions_array),
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
        print(path)
        state_dict = torch.load(path, map_location='cpu')
    
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        compatible_state_dict = {}
        for k, v in state_dict.items():
            k = k[4:]
            compatible_state_dict[k] = v

        net.load_state_dict(compatible_state_dict)

    return net



def main(classes_list):
    args = parser.parse_args()
    # train_opt = TrainOptions().parse()
    # net = EventNetwork(encoder_name='resnet101', num_classes=args.num_classes).cuda()
    # net.eval()
    # net = EventCnnLstm(encoder_name='resnet101', num_classes=23).cuda()
    # net.eval()
    net = MTResnetAggregate(args).cuda()
    # print(net)
    net = load_model(net, args.model_path)
    net.eval()
    # args = parser.parse_args()

    # Setup model
    # print('creating and loading the model...')
    # state = torch.load(args.model_path, map_location='cpu')
    # # args.num_classes = state['num_classes']
    # model = create_model(args).cuda()
    # model.load_state_dict(state['model'], strict=True)
    # model.eval()
    # classes_list = np.array(list(state['idx_to_class'].values()))
    # print('Class list:', classes_list)
    
    val_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        # T.Normalize(
        #     mean=(0.485, 0.456, 0.406),
        #     std=(0.229, 0.224, 0.225)
        # ),

    ])
    
    
    ds = AlbumsDataset(data_path='../CUFED/images',album_list = args.album_list,
                         transforms=val_transform, args=args)
    
    
    dataloader = data.DataLoader(ds, batch_size= 1, num_workers= 4, shuffle=False)
    acc = validate_model(net, dataloader, args.threshold)
    print(acc)

    # np_output = output.cpu().detach().numpy()
    
    
    # # np_output[np_output >= 0.5] = 1
    # # np_output[np_output < 0.5] = 0
    # # predict_label = np.where(np_output == 1)[1]
    # # print(predict_label)
    # # batch = tensor_batch.permute(0,2, 3, 1).cpu()
    # # plt.figure(figsize=(len(batch), len(batch)))
    # # for i in range(len(batch)):
    # #     plt.subplot(6, 6, i+1)
    # #     plot_image(i, np_output[i], event, batch, classes_list)
        
    # # plt.tight_layout()
    # # plt.savefig('result.png')
    # print(np_output)

    # idx_sort = np.argsort(-np_output)
    # # Top-k
    # detected_classes = np.array(classes_list)[idx_sort][: args.top_k]
    # scores = np_output[idx_sort][: args.top_k]

    # # Threshold
    # idx_th = scores > args.threshold
    # print(detected_classes[idx_th], scores[idx_th])
        
    # tags, confs = inference(tensor_batch, net, classes_list, args)

    # Visualization
    # display_image(montage, tags, 'result.jpg', os.path.join(
    #     args.path_output, args.album_path).replace("./albums", ""))
    
    # idx_sort = np.argsort(-np_output)
    # # # Top-k
    # detected_classes = np.array(classes_list)[idx_sort][:,:args.top_k]
    # print(detected_classes.shape)
    # scores = np.sort(np_output)[:,::-1][:,:args.top_k]
    # print(scores.shape)
    

    # # # Threshold
    # idx_th = scores >= 0.5
    # print(detected_classes[idx_th])
    # print(scores[idx_th])

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
    
if __name__ == '__main__':
    classes_list = ['Architecture', 'BeachTrip', 'Birthday', 'BusinessActivity',
                  'CasualFamilyGather', 'Christmas', 'Cruise', 'Graduation', 'GroupActivity',
                  'Halloween', 'Museum', 'NatureTrip', 'PersonalArtActivity',
                  'PersonalMusicActivity', 'PersonalSports', 'Protest', 'ReligiousActivity',
                  'Show', 'Sports', 'ThemePark', 'UrbanTrip', 'Wedding', 'Zoo']
    main(classes_list)
