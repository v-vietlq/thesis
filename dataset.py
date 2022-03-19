from PIL import Image
import torch.utils.data as data
import itertools
import os
import numpy as np
import argparse
import json
import torch
from torch.utils.data import sampler
from datasets.samplers import *
from functools import partial
from datasets.augmentations.generate_transforms import generate_validation_transform
from torchvision.transforms import transforms as T
import random
from torchvision.datasets import ImageFolder


def fast_collate(batch, clip_length=None):
  im1, im2,score1, score2
  batch_size = len(targets)
  dims = (batch[0][0].shape[2], batch[0][0].shape[0], batch[0][0].shape[1])  # HWC to CHW
  tensor_uint8_CHW = torch.empty((batch_size, *dims), dtype=torch.uint8)
  for i in range(batch_size):
    tensor_uint8_CHW[i] = torch.from_numpy(batch[i][0]).permute(2, 0, 1)  # # HWC to CHW
    # tensor_uint8_CHW[i] = batch[i][0].permute(2, 0, 1)  # # HWC to CHW # (Added)
  targets = targets.view(batch_size // clip_length, clip_length, -1)[:, 0]
  return tensor_uint8_CHW.float(), targets  # , extra_data

def fast_collate_1(batch,clip_length):

    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[4] for target in batch], dtype=torch.int64)
    batch_size = len(targets)
    w = imgs[0].shape[1]
    h = imgs[0].shape[2]
    tensor = torch.zeros( (len(imgs), 3, h, w), dtype=torch.float)
    for i, img in enumerate(imgs):
        tensor[i] += img
    targets = targets.view(batch_size // clip_length, clip_length, -1)[:, 0]
    return tensor, targets

def default_loader(path):
    img = Image.open(path)
    return img.convert('RGB')


def get_impaths_from_dir(dirpath, album_list, args=None):
    impaths = []
    labels = []
    scores = []

    # albums = os.listdir(dirpath)
    albums = np.loadtxt(album_list, dtype='str', delimiter='\n')

    cls_dict = json.load(open(args.event_type_pth))
    scores_dict = json.load(open(args.image_importance_pth))

    for album in albums:
        cls = cls_dict[album]
        scores_album = scores_dict[album]
        scores_per_album = []
        # if len(cls) > 1:
        #     continue
        imgs_album = os.listdir(os.path.join(dirpath, album))
        labels_album = [cls] * len(imgs_album)
        for img in imgs_album:
            name = img.replace('.jpg', '')
            path = os.path.join(album, name)
            for score in scores_album:
                if path == score[1]:
                    scores_per_album.append(score[2])
        scores.append(scores_per_album)
        impaths.append([os.path.join(album, img) for img in imgs_album])
        labels.append(labels_album)

    impaths = list(itertools.chain.from_iterable(impaths))  # flatten
    labels = list(itertools.chain.from_iterable(labels))  # flatten
    scores = list(itertools.chain.from_iterable(scores))

    normailize_scores = (scores - np.min(scores)) / \
        (np.max(scores) - np.min(scores))

    labels_str = ['Architecture', 'BeachTrip', 'Birthday', 'BusinessActivity',
                  'CasualFamilyGather', 'Christmas', 'Cruise', 'Graduation', 'GroupActivity',
                  'Halloween', 'Museum', 'NatureTrip', 'PersonalArtActivity',
                  'PersonalMusicActivity', 'PersonalSports', 'Protest', 'ReligiousActivity',
                  'Show', 'Sports', 'ThemePark', 'UrbanTrip', 'Wedding', 'Zoo']
    classes_to_idx = {}
    for i, cls in enumerate(labels_str):
        classes_to_idx[cls] = i

    if not(args is None) and hasattr(args, 'num_classes'):
        num_cls = args.num_classes
    else:
        num_cls = len(labels_str)
    # labels_idx = [classes_to_idx[lbl]  for lbl in labels]

    lbls = []
    for lbl in labels:
        labels_onehot = num_cls * [0]
        for lb in lbl:
            labels_onehot[classes_to_idx[lb]] = 1.0
        lbls.append(labels_onehot)

    return impaths, lbls, normailize_scores


class DatasetFromList(data.Dataset):
    """From List dataset."""

    def __init__(self, root=None, album_list=None, impaths=None, labels=None, scores=None,
                 transform=None, target_transform=None,
                 loader=default_loader, args=None):
        """
        Args:

            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if (impaths is None):
            impaths, labels, scores = get_impaths_from_dir(
                root, album_list, args)

        self.root = root
        # self.classes = idx_to_class
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.samples = tuple(zip(impaths, labels, scores))
        if not(scores is None):
            self.scores = scores

    def __getitem__(self, index):
        impath, target, score = self.samples[index]
        print(impath)
        img = self.loader(os.path.join(self.root, impath))
        if self.transform is not None:
            img = self.transform(img)
        # if self.target_transform is not None:
        # target = torch.from_numpy(np.array(target))
        return img, target, score

    def __len__(self):
        return len(self.samples)


class AlbumsDataset(data.Dataset):
    def __init__(self, data_path, album_list, transforms, args=None) -> None:
        super().__init__()
        self.args = args
        self.data_path = data_path
        self.albums = np.loadtxt(album_list, dtype='str', delimiter='\n')
        self.transforms = transforms
        self.cls_dict = json.load(open(args.event_type_pth))
        # self.scores_dict = json.load(open(args.image_importance_pth))

        self.labels_str = ['Architecture', 'BeachTrip', 'Birthday', 'BusinessActivity',
                           'CasualFamilyGather', 'Christmas', 'Cruise', 'Graduation', 'GroupActivity',
                           'Halloween', 'Museum', 'NatureTrip', 'PersonalArtActivity',
                           'PersonalMusicActivity', 'PersonalSports', 'Protest', 'ReligiousActivity',
                           'Show', 'Sports', 'ThemePark', 'UrbanTrip', 'Wedding', 'Zoo']

        self.num_cls = len(self.labels_str)
        self.classes_to_idx = {}
        for i, cls in enumerate(self.labels_str):
            self.classes_to_idx[cls] = i

    def _read_album(self, path):
        frames = []
        files = os.listdir(path)
        # album_name = args.album_path.rsplit('/', 1)[1]
        # cls_dict = json.load(open(args.event_type_pth))
        # event = cls_dict[album_name]
        n_files = len(files)
        # items = range(0, n_files)
        # idx_fetch = choices(items, k = self.args.album_clip_length)

        idx_fetch = np.linspace(
            0, n_files-1, self.args.album_clip_length, dtype=int)
        for i, id in enumerate(idx_fetch):
            im = Image.open(os.path.join(path, files[id])).convert('RGB')
            im = self.transforms(im)
            # im_resize = im.resize((224, 224))
            # np_img = np.array(im_resize, dtype=np.uint8)
            # im = torch.from_numpy(np_img).float() / 255.0
            frames.append(im)

        return torch.stack(frames)

    def __getitem__(self, index):
        album = self.albums[index]

        labels = self.cls_dict[album]
        labels_onehot = self.num_cls * [0]
        for lb in labels:
            labels_onehot[self.classes_to_idx[lb]] = 1

        return self._read_album(os.path.join(self.data_path, album)), torch.tensor(labels_onehot)

    def __len__(self):
        return len(self.albums)


class CUFEDImportanceDataset(data.Dataset):
    def __init__(self, data_path, album_list, transforms, args=None) -> None:
        super(CUFEDImportanceDataset, self).__init__()
        self.args = args
        self.data_path = data_path
        self.albums = np.loadtxt(album_list, dtype='str', delimiter='\n')
        self.transforms = transforms
        self.cls_dict = json.load(open(args.event_type_pth))

        self.labels_str = ['Architecture', 'BeachTrip', 'Birthday', 'BusinessActivity',
                           'CasualFamilyGather', 'Christmas', 'Cruise', 'Graduation', 'GroupActivity',
                           'Halloween', 'Museum', 'NatureTrip', 'PersonalArtActivity',
                           'PersonalMusicActivity', 'PersonalSports', 'Protest', 'ReligiousActivity',
                           'Show', 'Sports', 'ThemePark', 'UrbanTrip', 'Wedding', 'Zoo']

        self.num_cls = len(self.labels_str)
        self.classes_to_idx = {}
        for i, cls in enumerate(self.labels_str):
            self.classes_to_idx[cls] = i

        self.data = ImageFolder(data_path, transforms)

        self.scores_dict = json.load(open(args.image_importance_pth))

        # print(self.scores_dict.values())
        self.scores = {img[1]: img[2]
                       for imgs in self.scores_dict.values() for img in imgs}

        self.index_to_classes = {v: k for k,
                                 v in self.data.class_to_idx.items()}

        self.max_score = 2
        self.min_score = -2

    def __getitem__(self, index):
        img0_tuple = self.data.imgs[index]
        while True:
            # keep looping till the same class image is found
            img1_tuple = random.choice(self.data.imgs)
            if img0_tuple[1] == img1_tuple[1]:
                break
        album = self.index_to_classes[img0_tuple[1]]

        # get label
        labels = self.cls_dict[album]
        labels_onehot = self.num_cls * [0]
        for lb in labels:
            labels_onehot[self.classes_to_idx[lb]] = 1

        # get score
        path1 = '/'.join(img0_tuple[0].rsplit("/", 2)[-2:]).split('.')[0]
        path2 = '/'.join(img1_tuple[0].rsplit("/", 2)[-2:]).split('.')[0]
        score_img1 = (self.scores[path1] - self.min_score) / \
            (self.max_score - self.min_score)
        score_img2 = (self.scores[path2] - self.min_score) / \
            (self.max_score - self.min_score)
        # score_img2

        im1 = Image.open(img0_tuple[0]).convert('RGB')
        im2 = Image.open(img1_tuple[0]).convert('RGB')
        # print('--------')
        # print(img0_tuple[0], score_img1)
        # print(img1_tuple[0], score_img2)
        # print(labels_onehot)
        # print('--------\n')
        if self.transforms is not None:
            im1 = self.transforms(im1)
            im2 = self.transforms(im2)

        # print(class_id)
        # print(self.data.imgs[class_id])

        return im1, im2, score_img1, score_img2, labels_onehot

    def __len__(self):
        return len(self.data.imgs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='PETA: Photo album Event recognition using Transformers Attention.')
    parser.add_argument('--model_path', type=str,
                        default='./models_local/peta_32.pth')
    parser.add_argument('--album_path', type=str,
                        default='./albums/Graduation/0_92024390@N00')
    parser.add_argument('---album_list', type=str,
                        default='filenames/test.txt')
    # /Graduation') # /0_92024390@N00')
    parser.add_argument('--event_type_pth', type=str,
                        default='../CUFED_split/event_type.json')
    parser.add_argument('--image_importance_pth', type=str,
                        default='../CUFED_split/image_importance.json')
    parser.add_argument('--val_dir', type=str, default='./albums')
    parser.add_argument('--num_classes', type=int, default=23)
    parser.add_argument('--model_name', type=str, default='mtresnetaggregate')
    parser.add_argument('--transformers_pos', type=int, default=1)
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--transform_type', type=str, default='squish')
    parser.add_argument('--album_sample', type=str, default='uniform_ordered')
    parser.add_argument('--dataset_path', type=str, default='./data/ML_CUFED')
    parser.add_argument('--dataset_type', type=str, default='ML_CUFED')
    parser.add_argument('--path_output', type=str, default='./outputs')
    parser.add_argument('--use_transformer', type=int, default=1)
    parser.add_argument('--album_clip_length', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--top_k', type=int, default=3)
    parser.add_argument('--threshold', type=float, default=0.85)
    parser.add_argument('--remove_model_jit', type=int, default=None)
    args = parser.parse_args()

    val_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),

    ])

    ds = CUFEDImportanceDataset(data_path='../CUFED/images', album_list=args.album_list,
                                transforms=val_transform, args=args)

    val_sampler = OrderSampler(ds, args=args)
    collate_fn = lambda b: fast_collate_1(b, args.album_clip_length)

    dataloader = data.DataLoader(ds,batch_size =128, num_workers=4,sampler = val_sampler, shuffle=False, collate_fn=collate_fn)
    for i,(im1, label) in enumerate(dataloader):
      if (i+1) % 3 == 0:
        # print(label[64:96])
        # print(label[96:128])
        print(label)
        break

   

    # valid_dl_pytorch = torch.utils.data.DataLoader(
    #     ds, batch_size=args.batch_size, shuffle=False, pin_memory=True, sampler = val_sampler,
    #     num_workers=args.num_workers, drop_last=False, collate_fn=partial(fast_collate, clip_length=args.album_clip_length))
    # for i, (img, target, score) in enumerate(valid_dl_pytorch):
    #     print(target)
    #     print(img.shape)
    #     print(score.shape)
    #     if i ==2:
    #         break
