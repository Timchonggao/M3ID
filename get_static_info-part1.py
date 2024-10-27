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
 
        nodule_num = len(regions) # 结节数

        regions = [region for region in regions if region.bbox[0] >= 0 and region.bbox[1] >= 0 and region.bbox[2] <= mask.shape[0] and region.bbox[3] <= mask.shape[1]]

        # 取出最大的结节
        if len(regions) > 0:
            regions = sorted(regions, key=lambda x: x.area, reverse=True)
            region = regions[0]

            # 过滤掉把整个图像作为结节的情况
            if region.bbox[0] == 0 and region.bbox[1] == 0 and region.bbox[2] == mask.shape[0] and region.bbox[3] == mask.shape[1]:
                image_info = np.zeros((22,), dtype=np.float32)
                images_info.append(image_info)
                print(f"{mode} image {image_names[i]} skip the whole image as a nodule")
                continue

            if region.area < 30:
                print(f"{mode} image {image_names[i]} 检测到结节太小可能是误判, area is {region.area}")
                image_info = np.zeros((22,), dtype=np.float32)
                images_info.append(image_info)
                continue
        else:
            image_info = np.zeros((22,), dtype=np.float32)
            images_info.append(image_info)
            print(f"{mode} image {image_names[i]} 没有检测到结节")
            continue
        
        
        # 计算结节大小和形状信息
        shape_info = {
            "bbox": region.bbox, # 结节矩形框的坐标
            "centroid": region.centroid, # 结节的质心坐标

            "area": region.area, # 结节面积
            "perimeter": region.perimeter,  # 周长
            "equivdiameter" :region.equivalent_diameter, # 等效直径
            "compactness": (region.perimeter ** 2) / (4 * np.pi * region.area) if region.area > 0 else 0, # 判断结节形状是否规则
            "circularity": (4 * np.pi * region.area / region.perimeter**2) * (1 - 0.5 / (region.perimeter / (2 * np.pi) + 0.5))**2, # 圆度, 最大圆度值为 1,判断结节形状是否规则

            "extent": region.extent, # 扩张度, 区域中的像素数与边界框中总像素数的比率，以标量形式返回。计算方法为 Area 除以边界框的面积。
            
            "eccentricity": region.eccentricity, # 偏心率， 离心率是一个衡量椭圆形状的指标，其值在 0 到 1 之间。当离心率为 0 时，表示连接组件是一个完美的圆形。随着离心率的增加，形状变得更加椭圆。
            "major_axis_length": region.major_axis_length, # 长轴长度
            "minor_axis_length": region.minor_axis_length, # 短轴长度
            "aspect_ratio": region.major_axis_length/region.minor_axis_length, # 纵横比
            "orientation": region.orientation, # 方向, x 轴与椭圆长轴（该椭圆与区域具有相同的二阶矩）之间的角度，以标量形式返回。

            "convex_area": region.convex_area, # 凸多边形面积
            "convex_perimeter": measure.perimeter(region.convex_image), # 凸多边形周长
            "solidity": region.solidity, # 固体度, 凸包中区域内像素所占的比例，以标量形式返回。实度计算为 Area/ConvexArea。
            
            "moments": region.moments, # 矩
            "moments_central": region.moments_central, # 中心矩
            "moments_hu": region.moments_hu, # 霍夫变换
        }

        # 添加边缘规则性信息，清晰度分数，规则性
        region_mask_binary = (labeled_mask == region.label).astype(np.uint8) * 255
        # 获取轮廓上的点
        contours, _ = cv2.findContours(region_mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            contour = contours[0]
        else:
            contour = []

        edge_points = np.squeeze(contour) # (num_points, 2)
        # print(f"edge_points shape: {edge_points.shape}")

        # 边缘点坐标标准差
        edge_sd = np.std(edge_points, axis=0) 
        edge_points_std = edge_sd.mean()

        # Fourier变换后高频成分的能量
        fd = np.fft.fft(edge_points[:,0]) + 1j * np.fft.fft(edge_points[:,1])
        high_freq_energy = np.sum(np.abs(fd[len(fd)//2:]))

        # 计算曲率
        curvatures = []
        for i in range(len(edge_points)-2):
            p1, p2, p3 = edge_points[i], edge_points[i+1], edge_points[i+2]
            curvature = np.abs(p2[0] * p1[1] + p1[0] * p3[1] + p3[0] * p2[1] - p3[0] * p1[1] - p2[0] * p3[1] - p1[0] * p2[1]) / (np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2) * np.sqrt((p3[0]-p2[0])**2 + (p3[1]-p2[1])**2))
            curvatures.append(curvature)
        # 提取曲率统计特征
        curvature_mean = np.mean(curvatures) 
        curvature_max = np.max(curvatures)
        curvature_std = np.std(curvatures)

        convexity = np.mean(curvature)

        # 多边形曲率计算
        if len(contour) > 1:
            # 使用轮廓近似
            epsilon = 0.01 * cv2.arcLength(contour, True)

            # 计算近似轮廓上的曲率信息
            curvature = cv2.approxPolyDP(contour, epsilon, True)

            # 计算曲率的方差，表示平滑度
            convex_curvature_std= np.std(curvature)
        else:
            convex_curvature_std = 0


        # 计算边缘清晰度分数,计算检测到的边缘像素点数E与结节总像素数T的比值,E/T作为边界清晰度的量化指标。
        edge_clarity = cv2.Canny(region_mask_binary, 20, 150).sum() / region_mask_binary.sum()
        # print(f"边缘清晰度分数: {edge_clarity}")

        margin_info = {
            "edge_points_std": edge_points_std,
            "high_freq_energy": high_freq_energy,
            "curvature_std": curvature_std,
            "curvature_mean": curvature_mean,
            "curvature_max": curvature_max,
            "convexity": convexity,
            "convex_curvature_std": convex_curvature_std,
            "edge_clarity": edge_clarity
        }

        # 计算超声图像结节部分的像素信息，比如实性、囊性；
        nodule_mask = (labeled_mask == region.label).astype(np.uint8)
        nodule_image = image * nodule_mask[:, :, np.newaxis]
        nodule_image = cv2.cvtColor(nodule_image, cv2.COLOR_BGR2GRAY)

         # 计算结节像素平均值，这里使用np.mean()作为简化示例
        mean_pixel_value = np.mean(nodule_image[nodule_mask > 0])
        if mean_pixel_value < 100:
            composition = 1
        elif mean_pixel_value < 200:
            composition = 2
        else:
            composition = 3
    
        # 回声类型分析；


        # 钙化类型分析；
        

        # 将信息添加到数组
        image_info = [shape_info["area"], shape_info["perimeter"], shape_info["equivdiameter"], \
                    shape_info["compactness"], shape_info["circularity"], shape_info["extent"], shape_info["eccentricity"],\
                    shape_info["major_axis_length"], shape_info["minor_axis_length"], shape_info["aspect_ratio"], shape_info["orientation"], \
                    shape_info["convex_area"], shape_info["convex_perimeter"], shape_info["solidity"],\
                    margin_info["edge_points_std"], margin_info["high_freq_energy"], \
                    margin_info["curvature_std"], margin_info["curvature_mean"], margin_info["curvature_max"], \
                    margin_info["convexity"], margin_info["convex_curvature_std"], margin_info["edge_clarity"]]
                    # shape_info["moments"],shape_info["moments_central"], shape_info["moments_hu"]]
        image
        images_info.append(image_info)
    return np.array(images_info)

def get_clinical_information_multi_nodules(masks, images, image_names):
    batch_size = masks.shape[0]
    image_info = []
    for i in range(batch_size):
        # 保存每张图片的结节信息
        nodule_info = [] 
        # print(f"image_name: {image_names[i]}")
        mask = masks[i].cpu().numpy().transpose(1, 2, 0).squeeze().astype(np.uint8) # 这里需要将mask的维度变为二维
        # print(f"mask shape: {mask.shape}")

        image = images[i].cpu().numpy().transpose(1, 2, 0)

        # 对二值掩膜进行标签化
        labeled_mask = skimage_label(mask)
        regions = regionprops(labeled_mask)
        # 去掉最外围的边框，坐标为图像的左上角和右下角
        regions = [region for region in regions if region.bbox[0] >= 0 and region.bbox[1] >= 0 and region.bbox[2] <= mask.shape[0] and region.bbox[3] <= mask.shape[1] and region.area > 100]
        # print(f"结节个数: {len(regions)}")
        
        for region in regions:
            # 过滤掉把整个图像作为结节的情况
            if region.bbox[0] == 0 and region.bbox[1] == 0 and region.bbox[2] == mask.shape[0] and region.bbox[3] == mask.shape[1]:
                print("skip the whole image as a nodule")
                continue
            # print(f"label: {region.label}, area: {region.area}, bbox: {region.bbox}")

            # 添加结节大小和形状信息
            shape_info = {
                "bbox": region.bbox, # 结节矩形框的坐标
                "centroid": region.centroid, # 结节的质心坐标

                "area": region.area, # 结节面积
                "perimeter": region.perimeter,  # 周长
                "equivdiameter" :region.equivalent_diameter, # 等效直径
                "compactness": (region.perimeter ** 2) / (4 * np.pi * region.area) if region.area > 0 else 0, # 判断结节形状是否规则
                "circularity": (4 * np.pi * region.area / region.perimeter**2) * (1 - 0.5 / (region.perimeter / (2 * np.pi) + 0.5))**2, # 圆度, 最大圆度值为 1,判断结节形状是否规则

                "extent": region.extent, # 扩张度, 区域中的像素数与边界框中总像素数的比率，以标量形式返回。计算方法为 Area 除以边界框的面积。
                
                "eccentricity": region.eccentricity, # 偏心率， 离心率是一个衡量椭圆形状的指标，其值在 0 到 1 之间。当离心率为 0 时，表示连接组件是一个完美的圆形。随着离心率的增加，形状变得更加椭圆。
                "major_axis_length": region.major_axis_length, # 长轴长度
                "minor_axis_length": region.minor_axis_length, # 短轴长度
                "aspect_ratio": region.major_axis_length/region.minor_axis_length, # 纵横比
                "orientation": region.orientation, # 方向, x 轴与椭圆长轴（该椭圆与区域具有相同的二阶矩）之间的角度，以标量形式返回。

                "convex_area": region.convex_area, # 凸多边形面积
                "convex_perimeter": measure.perimeter(region.convex_image), # 凸多边形周长
                "solidity": region.solidity, # 固体度, 凸包中区域内像素所占的比例，以标量形式返回。实度计算为 Area/ConvexArea。
                
                "moments": region.moments, # 矩
                "moments_central": region.moments_central, # 中心矩
                "moments_hu": region.moments_hu, # 霍夫变换
            }

            # 添加边缘规则性信息，清晰度分数，规则性
            region_mask_binary = (labeled_mask == region.label).astype(np.uint8) * 255
            # 获取轮廓上的点
            contours, _ = cv2.findContours(region_mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour = contours[0] if contours else [] # contours的点的数量相比edges会比较少，但是可以计算其他的信息
            
            edge_points = np.squeeze(contour) # (num_points, 2)
            # print(f"edge_points shape: {edge_points.shape}")

            # 边缘点坐标标准差
            edge_sd = np.std(edge_points, axis=0) 
            edge_points_std = edge_sd.mean()

            # Fourier变换后高频成分的能量
            fd = np.fft.fft(edge_points[:,0]) + 1j * np.fft.fft(edge_points[:,1])
            high_freq_energy = np.sum(np.abs(fd[len(fd)//2:]))

            # 计算曲率
            curvatures = []
            for i in range(len(edge_points)-2):
                p1, p2, p3 = edge_points[i], edge_points[i+1], edge_points[i+2]
                curvature = np.abs(p2[0] * p1[1] + p1[0] * p3[1] + p3[0] * p2[1] - p3[0] * p1[1] - p2[0] * p3[1] - p1[0] * p2[1]) / (np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2) * np.sqrt((p3[0]-p2[0])**2 + (p3[1]-p2[1])**2))
                curvatures.append(curvature)
            # 提取曲率统计特征
            curvature_mean = np.mean(curvatures) 
            curvature_max = np.max(curvatures)
            curvature_std = np.std(curvatures)

            convexity = np.mean(curvature)

            # 多边形曲率计算
            if len(contour) > 1:
                # 使用轮廓近似
                epsilon = 0.01 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                # 计算近似轮廓上的曲率信息
                curvature = cv2.approxPolyDP(contour, epsilon, True)

                # 计算曲率的方差，表示平滑度
                convex_curvature_std= np.std(curvature)
            else:
                convex_curvature_std = 0


            # 计算边缘清晰度分数,计算检测到的边缘像素点数E与结节总像素数T的比值,E/T作为边界清晰度的量化指标。
            edge_clarity = cv2.Canny(region_mask_binary, 20, 150).sum() / region_mask_binary.sum()
            # print(f"边缘清晰度分数: {edge_clarity}")

            margin_info = {
                "edge_points_std": edge_points_std,
                "high_freq_energy": high_freq_energy,
                "curvature_std": curvature_std,
                "curvature_mean": curvature_mean,
                "curvature_max": curvature_max,
                "convexity": convexity,
                "convex_curvature_std": convex_curvature_std,
                "edge_clarity": edge_clarity
            }

            # 将信息添加到数组
            nodule_info.append({"margin_info": margin_info, "shape_info": shape_info})
        image_info.append(nodule_info)
    return image_info

def load_and_evaluate_model(backbone, dataloader):
    backbone.eval()
    all_metrics = np.empty((0, 47), dtype=np.float32)

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
        out_seg_metrics_tensor = torch.from_numpy(out_seg_metrics).float().cuda() # torch.Size([16, 22])
        # breakpoint()

        # 基于mask的分割结果计算结节的形态特征
        mask_binary = (mask * 255).to(torch.uint8)
        mask_metrics = get_clinical_information(mask_binary, img, image_name , mode = 'mask')
        mask_metrics_tensor = torch.from_numpy(mask_metrics).float().cuda() # torch.Size([16, 22])
        # breakpoint()

        # 将所有特征进行拼接
        image_metrics = torch.cat((mask_metrics_tensor, out_seg_metrics_tensor, prob, label.unsqueeze(1)), dim=1)
        all_metrics = np.vstack((all_metrics, image_metrics.cpu().detach().numpy())) # (batch_size, 47)
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

        trainset = tndt_dataset.TNDTDataset(mode='train', fold=fold, transform=composed_transforms_train, return_size=False)
        trainloader = DataLoader(trainset, batch_size=16, shuffle=True, num_workers=8, drop_last=True)

        valset = tndt_dataset.TNDTDataset(mode='val', fold=fold, transform=composed_transforms_val, return_size=False)
        valloader = DataLoader(valset, batch_size=1, shuffle=False, num_workers=2)

        testset = tndt_dataset.TNDTDataset(mode='test', transform=composed_transforms_test, return_size=False)
        testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)

        # save_dir = os.path.join('results', 'model-output', 'run-tn3k', 'swinunet' + '_lr' + str(0.001), 'fold' + str(fold))
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

        save_dir = 'image-static-info-part1/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        scaler = MinMaxScaler()

        trainset_metrics = load_and_evaluate_model(backbone, trainloader)
        # trainset_metrics = scaler.fit_transform(trainset_metrics)
        np.save('image-static-info-part1/all_metrics_fold{}_train.npy'.format(fold), trainset_metrics)
        
        valset_metrics = load_and_evaluate_model(backbone, valloader)
        # valset_metrics = scaler.fit_transform(valset_metrics)
        np.save('image-static-info-part1/all_metrics_fold{}_val.npy'.format(fold),  valset_metrics)

        testset_metrics = load_and_evaluate_model(backbone, testloader)
        # testset_metrics = scaler.fit_transform(testset_metrics)
        np.save('image-static-info-part1/all_metrics_fold{}_test.npy'.format(fold), testset_metrics)

if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()
# export CUDA_VISIBLE_DEVICES=7