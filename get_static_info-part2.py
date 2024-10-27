import os
from tqdm import tqdm
import time
import socket
import argparse
from datetime import datetime

import cv2
import numpy as np
from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import MinMaxScaler
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms


import math
import scipy
from skimage import measure
from skimage.measure import label as skimage_label, regionprops
import pywt
from skimage.filters import sobel, scharr, prewitt
from skimage.feature import canny
from skimage.filters import gabor_kernel
from skimage.feature import graycomatrix, graycoprops


# swin U-Net网络
from model.swinunet.swinuT import swinu_t as swinunet

# utils and dataloaders
from dataloaders import tndt_dataset
from dataloaders import tn3k_dataset
from dataloaders import custom_transforms as trforms


def get_clinical_information(masks, images, image_names, mode):
    batch_size = masks.shape[0]
    images_info = []
    for i in range(batch_size):
        image_info = []
        # print(f"image_name: {image_names[i]}")
        mask = masks[i].cpu().numpy().transpose(1, 2, 0).squeeze().astype(np.uint8) # 这里需要将mask的维度变为二维
        # print(f"mask shape: {mask.shape}")

        image = images[i].cpu().numpy().transpose(1, 2, 0)

        # 对二值掩膜进行标签化
        labeled_mask = skimage_label(mask)
        regions = regionprops(labeled_mask)
 
        regions = [region for region in regions if region.bbox[0] >= 0 and region.bbox[1] >= 0 and region.bbox[2] <= mask.shape[0] and region.bbox[3] <= mask.shape[1] and region.area > 20]
        # 计算结节的数量，过滤掉小于20像素的结节
        nodule_num = len(regions)

        # 取出最大的结节
        if len(regions) > 0:
            regions = sorted(regions, key=lambda x: x.area, reverse=True)
            region = regions[0]
            
            # 过滤掉把整个图像作为结节的情况
            if region.bbox[0] == 0 and region.bbox[1] == 0 and region.bbox[2] == mask.shape[0] and region.bbox[3] == mask.shape[1]:
                image_info = np.zeros((31,), dtype=np.float32)
                images_info.append(image_info)
                print(f"{mode} image {image_names[i]} skip the whole image as a nodule")
                continue

            if region.area < 30:
                print(f"{mode} image {image_names[i]} 检测到结节太小可能是误判, area is {region.area}")
                image_info = np.zeros((31,), dtype=np.float32)
                images_info.append(image_info)
                continue
        else:
            image_info = np.zeros((31,), dtype=np.float32)
            images_info.append(image_info)
            print(f"{mode} image {image_names[i]} 没有检测到结节")
            continue
        

        # 计算超声图像结节部分的像素信息，比如实性、囊性；
        nodule_mask = (labeled_mask == region.label).astype(np.uint8)
        nodule_image = image * nodule_mask[:, :, np.newaxis]
        nodule_image = cv2.cvtColor(nodule_image, cv2.COLOR_BGR2GRAY).astype(np.uint8)

         # 计算结节像素平均值，这里使用np.mean()作为简化示例
        mean_pixel_value = np.mean(nodule_image[nodule_mask > 0])
        if mean_pixel_value < 100:
            composition = 1
        elif mean_pixel_value < 200:
            composition = 2
        else:
            composition = 3
    
        # 计算边缘强度(Edge strength)
        edges = canny(nodule_image, sigma=3)
        edge_strength = np.mean(edges[edges > 0])

        # 计算边缘密度(Edge density)
        edge_density = np.sum(edges) / edges.size

        # 计算灰度共生矩阵(GLCM)
        glcm = graycomatrix(nodule_image, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
        # 从GLCM中提取纹理特征
        contrast = graycoprops(glcm, 'contrast')[0, 0] 
        dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
        energy = graycoprops(glcm, 'energy')[0, 0]
        correlation = graycoprops(glcm, 'correlation')[0, 0]

        # 计算小波变换
        wavelet = 'db4'  # 选择小波函数
        coeffs = pywt.wavedec2(nodule_image, wavelet, level=3)
        # 提取小波系数的统计特征
        wavelet_means = [np.mean(coeffs[0])]
        wavelet_stds = [np.std(coeffs[0])]
        for detail_level in range(1, len(coeffs)):
            wavelet_means.append(np.mean(coeffs[detail_level]))
            wavelet_stds.append(np.std(coeffs[detail_level]))

        # 定义Gabor滤波器
        freq = 0.6  # 频率
        theta = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # 方向角度
        gabor_kernels = [gabor_kernel(freq, theta=t) for t in theta]

        # 计算Gabor纹理特征
        gabor_features = []
        for kernel in gabor_kernels:
            filtered = np.real(np.fft.fft2(kernel, nodule_image.shape))
            gabor_features.append(np.mean(np.abs(filtered)))


        # 计算傅里叶变换
        f = np.fft.fft2(nodule_image)
        fshift = np.fft.fftshift(f)

        # 计算幅值谱
        magnitude_spectrum = np.abs(fshift)

        # 提取傅里叶变换特征
        fourier_mean = np.mean(magnitude_spectrum)
        fourier_std = np.std(magnitude_spectrum)
        fourier_energy = np.sum(magnitude_spectrum**2)

        # 计算Sobel梯度
        sobel_grad = sobel(nodule_image)
        sobel_mean = np.mean(sobel_grad)
        sobel_std = np.std(sobel_grad)

        # 计算Scharr梯度
        scharr_grad = scharr(nodule_image)
        scharr_mean = np.mean(scharr_grad)
        scharr_std = np.std(scharr_grad)

        # 计算Prewitt梯度
        prewitt_grad = prewitt(nodule_image)
        prewitt_mean = np.mean(prewitt_grad)
        prewitt_std = np.std(prewitt_grad)

        # 将信息添加到数组
        image_info = [nodule_num, composition, mean_pixel_value, edge_strength,
                      edge_density, contrast, dissimilarity, homogeneity,
                      energy, correlation,
                      wavelet_means[0], wavelet_stds[0], wavelet_means[1], wavelet_stds[1],
                      wavelet_means[2], wavelet_stds[2], wavelet_means[3], wavelet_stds[3], 
                      gabor_features[0], gabor_features[1], gabor_features[2], gabor_features[3],
                      fourier_mean, fourier_std, fourier_energy, sobel_mean, sobel_std,
                      scharr_mean, scharr_std, prewitt_mean, prewitt_std]

        images_info.append(image_info)
    return np.array(images_info)

def load_and_evaluate_model(backbone, dataloader):
    backbone.eval()
    all_metrics = np.empty((0, 65), dtype=np.float32)

    for ii, sample_batched in enumerate(dataloader):
        img = sample_batched['image'].cuda()
        label = sample_batched['label'].cuda()
        mask = sample_batched['mask'].cuda() # 这里的mask本身即是01 mask
        image_name = sample_batched['label_name']

        feats, out_seg = backbone.forward(img)
        prob = F.softmax(feats, dim=1)

        # 基于模型的分割结果计算结节的形态特征
        # 获取结节部位的掩码
        threshold = 0.5 
        out_seg_porb = torch.sigmoid(out_seg)
        out_seg_binary = ((out_seg_porb > threshold) * 255).to(torch.uint8)
        out_seg_metrics = get_clinical_information(out_seg_binary, img, image_name, mode = 'out_seg') # (batch_size, 22)
        out_seg_metrics_tensor = torch.from_numpy(out_seg_metrics).float().cuda() # torch.Size([16, 31])
        # breakpoint()
        # 基于mask的分割结果计算结节的形态特征
        mask_binary = (mask * 255).to(torch.uint8)
        mask_metrics = get_clinical_information(mask_binary, img, image_name , mode = 'mask')
        mask_metrics_tensor = torch.from_numpy(mask_metrics).float().cuda() # torch.Size([16, 31])
        # breakpoint()
        # 将所有特征进行拼接
        image_metrics = torch.cat((mask_metrics_tensor, out_seg_metrics_tensor, prob, label.unsqueeze(1)), dim=1)
        all_metrics = np.vstack((all_metrics, image_metrics.cpu().detach().numpy())) # (batch_size, 65)
        print('all_metrics shape:', all_metrics.shape)
        # breakpoint()
    return all_metrics

def main():
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    torch.backends.cudnn.deterministic = True

    composed_transforms_train = transforms.Compose([
        trforms.FixedResize(size=(224 + 8, 224 + 8)),
        trforms.RandomCrop(size=(224, 224)),
        trforms.RandomHorizontalFlip(),
        trforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        trforms.ToTensor()])

    composed_transforms_val = transforms.Compose([
        trforms.FixedResize(size=(224, 224)),
        trforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        trforms.ToTensor()])

    composed_transforms_test = transforms.Compose([
        trforms.FixedResize(size=(224, 224)),
        trforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        trforms.ToTensor()])
    
    backbone = swinunet(pretrained=False)

    for fold in range(5):
        # set dataset and data loader
        # trainset = tn3k_dataset.TN3KDataset(mode='train', fold=fold, transform=composed_transforms_train, return_size=False, Balance=True)
        # trainloader = DataLoader(trainset, batch_size=16, shuffle=True, num_workers=8, drop_last=True)

        # valset = tn3k_dataset.TN3KDataset(mode='val', fold=fold, transform=composed_transforms_val, return_size=False, Balance=False)
        # valloader = DataLoader(valset, batch_size=1, shuffle=False, num_workers=2)

        # testset = tn3k_dataset.TN3KDataset(mode='test', transform=composed_transforms_test, return_size=False, Balance=False)
        # testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)

        # save_dir = os.path.join('results', 'model-output', 'run-tn3k', 'swinunet' + '_lr' + str(0.001), 'fold' + str(fold))
        
        trainset = tndt_dataset.TNDTDataset(mode='train', fold=fold, transform=composed_transforms_train, return_size=False)
        trainloader = DataLoader(trainset, batch_size=16, shuffle=True, num_workers=8, drop_last=True)

        valset = tndt_dataset.TNDTDataset(mode='val', fold=fold, transform=composed_transforms_val, return_size=False)
        valloader = DataLoader(valset, batch_size=1, shuffle=False, num_workers=2)

        testset = tndt_dataset.TNDTDataset(mode='test', transform=composed_transforms_test, return_size=False)
        testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)

        save_dir = os.path.join('results', 'model-output', 'run-tndt', 'swinunet' + '_lr' + str(0.001), 'fold' + str(fold))
        
        model_path = os.path.join(save_dir, 'models', 'best_backbone.pth')
        if os.path.exists(model_path):
            print('Evaluating model: ', model_path)
            backbone.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
            backbone.cuda()
            torch.cuda.set_device(device=0)
        else:
            print('Model not found: ', model_path)
            continue

        save_dir = 'image-static-info-part2/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        scaler = MinMaxScaler()

        trainset_metrics = load_and_evaluate_model(backbone, trainloader)
        # trainset_metrics = scaler.fit_transform(trainset_metrics)
        np.save('image-static-info-part2/all_metrics_fold{}_train.npy'.format(fold), trainset_metrics)
        
        valset_metrics = load_and_evaluate_model(backbone, valloader)
        # valset_metrics = scaler.fit_transform(valset_metrics)
        np.save('image-static-info-part2/all_metrics_fold{}_val.npy'.format(fold),  valset_metrics)

        testset_metrics = load_and_evaluate_model(backbone, testloader)
        # testset_metrics = scaler.fit_transform(testset_metrics)
        np.save('image-static-info-part2/all_metrics_fold{}_test.npy'.format(fold), testset_metrics)

if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()
