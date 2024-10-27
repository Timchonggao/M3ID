import os
import cv2
import json
import numpy as np
from PIL import Image

import torch
from torch.utils import data


def make_dataset(root, seed, balance=True):
    imgs = []
    img_labels = {}

    benign_num = 0
    malignant_num = 0

    # get label dict
    with open(root + 'label4trainval.csv', 'r') as f:
        lines = f.readlines()

    for idx in range(len(lines)):
        line = lines[idx]
        name, label = line.strip().split(',')
        img_labels[name] = label
    # get image path
    img_names = os.listdir(root + 'trainval-image/')
    for i in seed: # 通过json文件的seed选择训练集
        img_name = img_names[i] # n.jpg
        img = os.path.join(root + 'trainval-image/', img_name) # 路径
        mask = os.path.join(root + 'trainval-mask/', img_name)
        if int(img_labels[img_name]) == 1: # 恶性样本加权，训练集中恶性样本数量为974，良性样本数量为1905
            if balance:
                malignant_num += 2
                imgs.append((img, mask, img_labels[img_name]))
            else:
                malignant_num += 1
        elif int(img_labels[img_name]) == 0: # 良性样本
            benign_num += 1
        imgs.append((img, mask, img_labels[img_name])) # 图片路径，掩膜路径，标签
    print('train set benign_num:', benign_num,'malignant_num:', malignant_num)
    return imgs, benign_num, malignant_num

def make_validset(root, seed, balance):
    imgs = []
    img_labels = {}

    benign_num = 0
    malignant_num = 0

    # get label dict
    with open(root + 'label4trainval.csv', 'r') as f:
        lines = f.readlines()

    for idx in range(len(lines)):
        line = lines[idx]
        name, label = line.strip().split(',')
        img_labels[name] = label
    # get image path
    img_names = os.listdir(root + 'trainval-image/')
    for i in seed:        
        img_name = img_names[i]
        img = os.path.join(root + 'trainval-image/', img_name)
        mask = os.path.join(root + 'trainval-mask/', img_name)
        if int(img_labels[img_name]) == 1:
            malignant_num += 1
        elif int(img_labels[img_name]) == 0 :
            if balance:
                if benign_num < 207:
                    benign_num += 1
                else:
                    continue
            else:
                benign_num += 1
        imgs.append((img, mask, img_labels[img_name]))
    print('valid set benign_num:', benign_num,'malignant_num:', malignant_num)
    return imgs, benign_num, malignant_num

def make_testset(root, balance):
    imgs = []
    img_labels = {}

    benign_num = 0
    malignant_num = 0

    # get label dict
    with open(root + 'label4test.csv', 'r') as f:
        lines = f.readlines()

    for idx in range(0, len(lines)):
        line = lines[idx]
        name, label = line.strip().split(',')
        img_labels[name] = label

    # get image path
    img_names = os.listdir(root + 'test-image/')
    for img_name in img_names:
        img = os.path.join(root + 'test-image/', img_name)
        mask = os.path.join(root + 'test-mask/', img_name)
        if int(img_labels[img_name]) == 1:
            malignant_num += 1
        elif int(img_labels[img_name]) == 0:
            if balance:
                if benign_num < 236:
                    benign_num += 1
                else:
                    continue
            else:
                benign_num += 1
        imgs.append((img, mask, img_labels[img_name]))
    print('test set benign_num:', benign_num,'malignant_num:', malignant_num)
    return imgs, benign_num, malignant_num


class TN3KDataset(data.Dataset):
    def __init__(self, mode='train', transform=None, return_size=False, fold=0, Balance=False):
        self.mode = mode
        self.transform = transform
        self.return_size = return_size

        root = '/data3/gaochong/dataset/tn3k/datasets/tn3k/'
        
        
        trainvaltest = json.load(open(root + 'tn3k-trainval-fold' + str(fold) + '.json', 'r')) # 加载json文件，划分好的训练集、验证集、测试集
        
        if mode == 'train':
            imgs, train_benign_num, train_malignant_num = make_dataset(root, trainvaltest['train'], balance = Balance)
            self.benign_num = train_benign_num
            self.malignant_num = train_malignant_num
        elif mode == 'val':
            imgs, val_benign_num, val_malignant_num = make_validset(root, trainvaltest['val'], balance = Balance)
            self.benign_num = val_benign_num
            self.malignant_num = val_malignant_num
        elif mode == 'test':
            imgs, test_benign_num, test_malignant_num = make_testset(root, balance = Balance)
            self.test_benign_num = test_benign_num
            self.test_malignant_num = test_malignant_num

        self.imgs = imgs
        self.transform = transform
        self.return_size = return_size

    def __getitem__(self, item):
        
        image_path, mask_path, label = self.imgs[item]

        assert os.path.exists(image_path), ('{} does not exist'.format(image_path))
        assert os.path.exists(mask_path), ('{} does not exist'.format(mask_path))

        image = Image.open(image_path).convert('RGB')
        mask = np.array(Image.open(mask_path).convert('L'))
        if mask.max() != 0:
            mask = mask / mask.max()
        mask = Image.fromarray(mask.astype(np.uint8)) # 处理成了一个灰度图
        w, h = image.size
        size = (h, w)

        sample = {'image': image, 'mask':mask, 'label': int(label)}

        if self.transform:
            sample = self.transform(sample)
        # if self.return_size:
        #     sample['size'] = torch.tensor(size)

        label_name = os.path.basename(image_path)
        sample['label_name'] = label_name

        return sample

    def __len__(self):
        return len(self.imgs)