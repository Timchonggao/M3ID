import os
import csv
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms


from dataloaders import utils
from dataloaders import tndt_dataset
from dataloaders import tn3k_dataset
from dataloaders import custom_transforms as trforms
from dataloaders import custom_transforms_for_logonet as trforms_logo

from sklearn.metrics import roc_auc_score
from visualization.metrics import Metrics, evaluate

# import models
# Logonet模型
from torchvision.models import resnet18
from model.logonet.seg_lossed import *
from model.logonet.models import logonet # 基于resnet18的logonet模型，设计了一种多任务学习机制

# swin T系列网络
from model.swinnet.swinT import swin_t as swinnet # 单任务网络
from model.swinunet.swinuT import swinu_t as swinunet # 修改为多任务
from model.swinunet_refine.swinuT import swinu_t as swinunet_refine # 多任务网络，修改了分割任务的decoder concate setting
from model.swinunet_refine_1.swinuT import swinu_t as swinunet_refine_1 # 多任务网络，修改了分割任务的decoder concate setting,去掉首尾的concate
from model.swinunet_refine1.swinuT import swinu_t as swinunet_refine1 # 多任务网络，修改了decoder的unsample设置
from model.swinunet_refine2.swinuT import swinu_t as swinunet_refine2 # 基于refine的多任务网络，修改了最后出mask的upsample
from model.swinunet_refine3.swinuT import swinu_t as swinunet_refine3 # 基于refine的多任务网络，添加了TAD block
from model.swinunet_refine4.swinuT import swinu_t as swinunet_refine4 # 基于refine的多任务网络，设计了一个新的特征融合模块
from model.swinunet_refine4_1.swinuT import swinu_t as swinunet_refine4_1 # 基于自己设计的特征融合模块的多任务网络，debug
from model.swinunet_cis.swinuT import swinu_t as swinunet_cis # 多任务网络，增加了CIS模块

# 其他单任务模型
from torchvision.models import densenet121, convnext_tiny
from torchvision.models.vision_transformer import vit_b_16 as vit


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type=str, default='0')

    # Model setting
    parser.add_argument('-backbone', type=str, default='resnet18')
    parser.add_argument('-lr', type=float, default=1e-3)
    parser.add_argument('-input_size', type=int, default=224)
    parser.add_argument('-loss_func', type=str, default='ce')  # ce, superloss, curriloss

    # Training setting
    parser.add_argument('-seed', type=int, default=1234)
    parser.add_argument('-batch_size', type=int, default=16)
    parser.add_argument('-resume_epoch', type=int, default=0)
    parser.add_argument('-fold', type=int, default=0)
    parser.add_argument('-balance', action='store_true', help='启用详细输出')


    # Dataset setting
    parser.add_argument('-dataset', type=str, default='tn3k')
    return parser.parse_args()


def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if args.backbone == 'logonet':
        composed_transforms_ts = transforms.Compose([
            trforms_logo.FixedResize(size=(args.input_size, args.input_size),mask_size=(56, 56)),
            trforms_logo.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            trforms_logo.ToTensor()])
    else:
        composed_transforms_ts = transforms.Compose([
            trforms.FixedResize(size=(args.input_size, args.input_size)),
            trforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            trforms.ToTensor()])


    if args.dataset == 'tn3k':
        testset = tn3k_dataset.TN3KDataset(mode='test', transform=composed_transforms_ts, return_size=False, Balance=args.balance)
        testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)
    elif args.dataset == 'tndt':
        testset = tndt_dataset.TNDTDataset(mode='test', transform=composed_transforms_ts, return_size=False, Balance=args.balance)
        testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)
    acc = []
    presicion = [] 
    recall = [] 
    F1 = [] 
    auc = []
    mean = []
    std = []
    
    out_seg = None
    seg_recall = [] 
    seg_specificity = []
    seg_precision = []
    seg_F1 = []
    seg_acc = []
    seg_iou = []
    seg_mae = []
    seg_dice = []
    seg_hd = []
    seg_auc = []
    seg_mean = []
    seg_std = []

    for fold in range(5):
        if args.backbone == 'resnet':
            backbone = resnet18(pretrained=False)
            backbone.fc = nn.Linear(in_features=backbone.fc.in_features, out_features=2)
        elif args.backbone == 'densenet':
            backbone = densenet121(pretrained=False)
            backbone.classifier = nn.Linear(1024, 2)
        elif args.backbone == 'convnext':
            backbone = convnext_tiny(pretrained=False)
            backbone.classifier = nn.Sequential(nn.Flatten(1), nn.Linear(768, 2))
        elif args.backbone == 'vit':
            backbone = vit(weights='IMAGENET1K_V1')
            backbone.heads = nn.Linear(in_features=768, out_features=2)
        elif args.backbone == 'logonet':
            backbone = logonet.logonet18(pretrained=False)
        elif args.backbone == 'swinnet':
            backbone = swinnet(pretrained=False)
        elif args.backbone == 'swinunet':
            backbone = swinunet(pretrained=False)
        elif args.backbone == 'swinunet_refine':
            backbone = swinunet_refine(pretrained=False)
        elif args.backbone == 'swinunet_refine_1':
            backbone = swinunet_refine_1(pretrained=False)
        elif args.backbone == 'swinunet_refine1':
            backbone = swinunet_refine1(pretrained=False)
        elif args.backbone == 'swinunet_refine2':
            backbone = swinunet_refine2(pretrained=False)
        elif args.backbone == 'swinunet_refine3':
            backbone = swinunet_refine3(pretrained=False)
        elif args.backbone == 'swinunet_refine4':
            backbone = swinunet_refine4(pretrained=False)
        elif args.backbone == 'swinunet_refine4_1':
            backbone = swinunet_refine4_1(pretrained=False)
        elif args.backbone == 'swinunet_cis':
            pretrained_path = '/data3/gaochong/project/M3ID/results/model-output/run-tn3k/swinunet_lr0.001/fold{}/models/best_backbone.pth'.format(fold)
            backbone = swinunet_cis(pretrained=True, pretrained_path=pretrained_path)
            static_data_dir = '/data3/gaochong/project/M3ID/static_info'
            static_data_path = os.path.join(static_data_dir, 'all_metrics_fold' + str(fold) + '_test.npy')
            loaded_data = np.load(static_data_path)
            shape_metrics = loaded_data[:, 2:-3]
            # 截取shape_metrics指定列
            shape_metrics = shape_metrics[:, [2, 3, 4, 6, 7]]

            maskshape_mean = np.mean(shape_metrics, axis=0)
            maskshape_std = np.std(shape_metrics, axis=0)
        else:
            raise NotImplementedError

        save_dir = os.path.join(f'results/model-output/run-{args.dataset}', args.backbone + '_lr' + str(args.lr), 'fold' + str(fold))
        model_path = os.path.join(save_dir, 'models', 'best_backbone.pth')
        print('model_path is', model_path)
        if not os.path.exists(model_path):
            print('model not exist')
            continue

        backbone.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
        torch.cuda.set_device(device=0)
        backbone.cuda()
        backbone.eval()

        TP, FP, FN, TN = 0, 0, 0, 0

        # class metrics
        labels = []
        preds = []
        acc_good = acc_bad = 0

        metrics = Metrics(['precision', 'recall', 'specificity', 'F1_score', 'auc', 'acc', 'iou', 'dice', 'mae', 'hd'])
        total_iou = 0

        for sample_batched in tqdm(testloader):
            img = sample_batched['image'].cuda()
            label = sample_batched['label'].cuda()
            mask = sample_batched['mask'].cuda().float()

            if args.backbone in ['logonet', 'swinunet', 'swinunet_refine', 'swinunet_refine_1',
                                'swinunet_refine1', 'swinunet_refine2', 'swinunet_refine3', 
                                 'swinunet_refine4', 'swinunet_refine4_1', 'swinunet_final', ]:
                feats, out_seg = backbone.forward(img)
            else:
                if args.backbone in ['swinunet_cis']:
                    feats = backbone.forward(img, maskshape_mean, maskshape_std)
                else:
                    feats = backbone.forward(img)

            probability = F.softmax(feats, dim=1)
            pred = torch.argmax(feats, dim=1, keepdim=False)
            # breakpoint()

            labels.append(label.cpu().detach().numpy())
            preds.append(probability[0][1].cpu().detach().numpy())
            if pred == label == 0:
                acc_good += 1
            if pred == label == 1:
                acc_bad += 1        
            if pred == label == 1:
                TP += 1
            if pred == label == 0:
                TN += 1
            if pred == 1 and label == 0:
                FP += 1
            if pred == 0 and label == 1:
                FN += 1
            
            if out_seg is not None:
                prob_pred = torch.sigmoid(out_seg)
                iou = utils.get_iou(prob_pred, mask)
                _precision, _recall, _specificity, _f1, _auc, _acc, _iou, _dice, _mae, _hd = evaluate(prob_pred, mask)
                metrics.update(recall=_recall, specificity=_specificity, precision=_precision,
                           F1_score=_f1, acc=_acc, iou=_iou, mae=_mae, dice=_dice, hd=_hd, auc=_auc)
                total_iou += iou

        preds = np.array(preds)
        labels = np.array(labels)
        curauc = roc_auc_score(labels, preds)
        curpresicion = TP / (TP + FP)
        currecall = TP / (TP + FN)
        curF1 = 2 * curpresicion * currecall / (curpresicion + currecall)
        curacc = 0.5 * (acc_good / testset.test_benign_num + acc_bad / testset.test_malignant_num) # 加权正确率,差不多
        # curacc = (acc_good + acc_bad) / len(teset) # 正确率

        # breakpoint()
        accstr = str(round(curacc, 3))
        presicionstr = str(round(curpresicion, 3))
        recallstr = str(round(currecall, 3))
        F1str = str(round(curF1, 3))
        aucstr = str(round(curauc, 3))

        acc.append(curacc)
        presicion.append(curpresicion) 
        recall.append(currecall) 
        F1.append(curF1) 
        auc.append(curauc) 

        print(f'{args.dataset} test partition', 'acc:'+accstr+' precision:'+presicionstr+' recall:'+recallstr+' f1:'+F1str+' auc:'+aucstr)
        breakpoint()
        if total_iou != 0:
            metrics_result = metrics.mean(len(testloader))
            print('recall: %.4f, specificity: %.4f, precision: %.4f, F1_score:%.4f, acc: %.4f, iou: %.4f, mae: %.4f, dice: %.4f, hd: %.4f, auc: %.4f'
                % (metrics_result['recall'], metrics_result['specificity'], metrics_result['precision'],
                metrics_result['F1_score'],
                metrics_result['acc'], metrics_result['iou'], metrics_result['mae'], metrics_result['dice'],
                metrics_result['hd'], metrics_result['auc']))
            seg_recall.append(metrics_result['recall'])
            seg_specificity.append(metrics_result['specificity'])
            seg_precision.append(metrics_result['precision'])
            seg_F1.append(metrics_result['F1_score'])
            seg_acc.append(metrics_result['acc'])
            seg_iou.append(metrics_result['iou'])
            seg_mae.append(metrics_result['mae'])
            seg_dice.append(metrics_result['dice'])
            seg_hd.append(metrics_result['hd'])
            seg_auc.append(metrics_result['auc'])

    csvFile = open(f"results/test-result/{args.dataset}-test.csv", "a+")
    writer = csv.writer(csvFile)
    name = args.backbone + "-" + args.loss_func + '-lr' + str(args.lr)
    head = ["acc", "presicion", "recall", "F1", "AUC", name]
    writer.writerow(head)

    mean.append(round(np.mean(acc),4))
    mean.append(round(np.mean(presicion),4))
    mean.append(round(np.mean(recall),4))
    mean.append(round(np.mean(F1),4))
    mean.append(round(np.mean(auc),4))
    writer.writerow(mean)

    std.append(round(np.std(acc),4))
    std.append(round(np.std(presicion),4))
    std.append(round(np.std(recall),4))
    std.append(round(np.std(F1),4))
    std.append(round(np.std(auc),4))
    writer.writerow(std)

    csvFile.close()

    if total_iou != 0:
        csvFile = open(f"results/test-result/{args.dataset}-test.csv", "a+")
        writer = csv.writer(csvFile)
        name = args.backbone + "-" + args.loss_func + '-lr' + str(args.lr) + '-seg'
        head = ["recall", "specificity", "precision", "F1_score", "acc", "iou", "mae", "dice", "hd", "auc", name]
        writer.writerow(head)

        seg_mean.append(round(np.mean(seg_recall),4))
        seg_mean.append(round(np.mean(seg_specificity),4))
        seg_mean.append(round(np.mean(seg_precision),4))
        seg_mean.append(round(np.mean(seg_F1),4))
        seg_mean.append(round(np.mean(seg_acc),4))
        seg_mean.append(round(np.mean(seg_iou),4))
        seg_mean.append(round(np.mean(seg_mae),4))
        seg_mean.append(round(np.mean(seg_dice),4))
        seg_mean.append(round(np.mean(seg_hd),4))
        seg_mean.append(round(np.mean(seg_auc),4))
        writer.writerow(seg_mean)

        seg_std.append(round(np.std(seg_recall),4))
        seg_std.append(round(np.std(seg_specificity),4))
        seg_std.append(round(np.std(seg_precision),4))
        seg_std.append(round(np.std(seg_F1),4))
        seg_std.append(round(np.std(seg_acc),4))
        seg_std.append(round(np.std(seg_iou),4))
        seg_std.append(round(np.std(seg_mae),4))
        seg_std.append(round(np.std(seg_dice),4))
        seg_std.append(round(np.std(seg_hd),4))
        seg_std.append(round(np.std(seg_auc),4))
        writer.writerow(seg_std)

        csvFile.close()

if __name__ == '__main__':
    args = get_arguments()
    main(args)
