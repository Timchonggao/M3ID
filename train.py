import os
import time
import socket
import argparse
from datetime import datetime

import numpy as np
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms

# import models
# single task models
from torchvision.models import convnext_tiny, densenet121
from torchvision.models.vision_transformer import vit_b_16 as vit

# Logonet模型
from torchvision.models import resnet18
from model.logonet.seg_lossed import *
from model.logonet.models import logonet # 基于resnet18的logonet模型，设计了一种多任务学习机制

# swin T系列网络
from model.swinnet.swinT import swin_t as swinnet # 单任务网络
from model.swinunet.swinuT import swinu_t as swinunet # 多任务网络，增加了分割任务
from model.swinunet_refine.swinuT import swinu_t as swinunet_refine # 多任务网络，修改了分割任务的decoder concate setting，完善unet的结构
from model.swinunet_refine_1.swinuT import swinu_t as swinunet_refine_1 # 多任务网络，修改了分割任务的decoder concate setting,去掉首尾的concate
from model.swinunet_refine1.swinuT import swinu_t as swinunet_refine1 # 多任务网络，修改了decoder的unsample设置
from model.swinunet_refine2.swinuT import swinu_t as swinunet_refine2 # 基于refine的多任务网络，修改了最后出mask的upsample
from model.swinunet_refine3.swinuT import swinu_t as swinunet_refine3 # 基于refine的多任务网络，添加了TAD block
from model.swinunet_refine4.swinuT import swinu_t as swinunet_refine4 # 基于refine的多任务网络，设计了一个新的特征融合模块
from model.swinunet_refine4_1.swinuT import swinu_t as swinunet_refine4_1 # 基于自己设计的特征融合模块的多任务网络，debug
from model.swinunet_cis.swinuT import swinu_t as swinunet_cis # 多任务网络，增加了CIS模块

# # Vision Mamba 系列模型
# from model.vmamba.vision_mamba import Mambanet as MambaNet
# from model.vmamba_unet.vision_mamba import MambaUnet 



# utils and dataloaders
import utils
from dataloaders import tndt_dataset
from dataloaders import tn3k_dataset
from dataloaders import custom_transforms as trforms
from dataloaders import custom_transforms_for_logonet as trforms_logo


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type=str, default='0')

    # Dataset setting
    parser.add_argument('-dataset', type=str, default='tn3k')

    # Model setting
    parser.add_argument('-backbone', type=str, default='resnet18')
    parser.add_argument('-loss_func', type=str, default='ce')
    parser.add_argument('-input_size', type=int, default=224)

    # Training setting
    parser.add_argument('-seed', type=int, default=1234)
    parser.add_argument('-batch_size', type=int, default=16)
    parser.add_argument('-nepochs', type=int, default=50)
    parser.add_argument('-resume_epoch', type=int, default=0)

    # Optimizer setting
    parser.add_argument('-lr', type=float, default=1e-3)
    parser.add_argument('-update_lr_every', type=int, default=10)
    parser.add_argument('-log_every', type=int, default=100)
    parser.add_argument('-fold', type=int, default=0)

    return parser.parse_args()


def evaluate(backbone, dataloader, epoch, fold, best_auc, logger_path, save_dir, modelname, mode = 'val'):
    backbone.eval()

    acc = 0.0
    TP, FP, FN, TN = 0, 0, 0, 0
    preds = []
    labels = []
    save_epoch = 5
    
    for ii, sample_batched in enumerate(dataloader):
        img, label = sample_batched['image'].cuda(), sample_batched['label'].cuda()
        if modelname in ['logonet', 'vmamba_unet', 'swinunet', 'swinunet_refine', 'swinunet_refine_1', 
                         'swinunet_refine1', 'swinunet_refine2', 'swinunet_refine3',
                          'swinunet_refine4','swinunet_refine4_1', 'swinunet_final']:
            feats, out_seg = backbone.forward(img)
        else:
            if modelname in ['swinunet_cis']:
                static_data_dir = '/data3/gaochong/project/M3ID/static_info' 
                if mode == 'val':
                    static_data_path = os.path.join(static_data_dir, 'all_metrics_fold' + str(fold) + '_val.npy')
                if mode == 'test':
                    static_data_path = os.path.join(static_data_dir, 'all_metrics_fold' + str(fold) + '_test.npy')
                loaded_data = np.load(static_data_path)
                breakpoint()
                shape_metrics = loaded_data[:, 2:-3]
                # 截取shape_metrics指定列
                shape_metrics = shape_metrics[:, [2, 3, 4, 6, 7]]

                maskshape_mean = np.mean(shape_metrics, axis=0)
                maskshape_std = np.std(shape_metrics, axis=0)

                feats = backbone.forward(img, maskshape_mean, maskshape_std)
            else:
                feats = backbone.forward(img)
        pred = torch.argmax(feats, dim=1, keepdim=False)
        prob = F.softmax(feats, dim=1)
        # breakpoint()
        
        labels.append(label.cpu().detach().numpy())
        preds.append(prob[0][1].cpu().detach().numpy())

        if pred == label:
            acc += 1
        if pred == label == 1:
            TP += 1
        if pred == label == 0:
            TN += 1
        if pred == 1 and label == 0:
            FP += 1
        if pred == 0 and label == 1:
            FN += 1

    preds = np.array(preds)
    labels = np.array(labels)
    auc = roc_auc_score(labels, preds)

    acc /= len(labels)
    presicion = TP / (TP + FP + 0.000001)
    recall = TPR = TP / (TP + FN + 0.000001)
    f1 = 2 * presicion * recall / (presicion + recall + 0.000001)

    if mode == 'val':
        if epoch >= save_epoch:
            backbone_save_path = os.path.join(save_dir, 'models', 'best_backbone.pth')
            if auc > best_auc:
                best_auc = auc
                torch.save(backbone.state_dict(), backbone_save_path)

        print('fold %d val epoch: %d, images: %d, acc: %.4f, auc: %.4f, f1: %.4f, presicion: %.4f, recall: %.4f \n' % \
        (fold, epoch, len(labels), acc, auc, f1, presicion, recall))
        with open(logger_path, 'a') as f:
            f.write('fold %d val epoch: %d, images: %d, acc: %.4f, auc: %.4f, f1: %.4f, presicion: %.4f, recall: %.4f \n' % \
            (fold, epoch, len(labels), acc, auc, f1, presicion, recall))
        
        backbone.train()
        return best_auc
    elif mode == 'test':
        print('fold %d test epoch: %d, images: %d, acc: %.4f, auc: %.4f, f1: %.4f, presicion: %.4f, recall: %.4f \n' % \
        (fold, epoch, len(labels), acc, auc, f1, presicion, recall))
        with open(logger_path, 'a') as f:
            f.write('fold %d test epoch: %d, images: %d, acc: %.4f, auc: %.4f, f1: %.4f, presicion: %.4f, recall: %.4f \n' % \
            (fold, epoch, len(labels), acc, auc, f1, presicion, recall))
        backbone.train()
        return 


def main(args):
    # set seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True # for reproducibility

    # set data loader transforms
    if args.backbone == 'logonet':
        composed_transforms_train = transforms.Compose([
            trforms_logo.FixedResize(size=(args.input_size + 8, args.input_size + 8), mask_size=(56, 56)),
            trforms_logo.RandomCrop(size=(args.input_size, args.input_size)),
            trforms_logo.RandomHorizontalFlip(),
            trforms_logo.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            trforms_logo.ToTensor()])

        composed_transforms_val = transforms.Compose([
            trforms_logo.FixedResize(size=(args.input_size, args.input_size),mask_size=(56, 56)),
            trforms_logo.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            trforms_logo.ToTensor()])
        
        composed_transforms_test = transforms.Compose([
            trforms_logo.FixedResize(size=(args.input_size, args.input_size),mask_size=(56, 56)),
            trforms_logo.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            trforms_logo.ToTensor()])
    
    else:
        composed_transforms_train = transforms.Compose([
            trforms.FixedResize(size=(args.input_size + 8, args.input_size + 8)),
            trforms.RandomCrop(size=(args.input_size, args.input_size)),
            trforms.RandomHorizontalFlip(),
            trforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            trforms.ToTensor()])

        composed_transforms_val = transforms.Compose([
            trforms.FixedResize(size=(args.input_size, args.input_size)),
            trforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            trforms.ToTensor()])
        
        composed_transforms_test = transforms.Compose([
            trforms.FixedResize(size=(args.input_size, args.input_size)),
            trforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            trforms.ToTensor()])
        
    for fold in range(args.fold, args.fold+1): # 每次只训练一个fold，这样可以在脚本中同时训练多个fold
        # set dataset and data loader
        if args.dataset == 'tn3k':
            trainset = tn3k_dataset.TN3KDataset(mode='train', fold=fold, transform=composed_transforms_train, return_size=False)
            trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True)

            valset = tn3k_dataset.TN3KDataset(mode='val', fold=fold, transform=composed_transforms_val, return_size=False)
            valloader = DataLoader(valset, batch_size=1, shuffle=False, num_workers=2)

            testset = tn3k_dataset.TN3KDataset(mode='test', transform=composed_transforms_test, return_size=False)
            testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)
        elif args.dataset == 'tndt':
            trainset = tndt_dataset.TNDTDataset(mode='train', fold=fold, transform=composed_transforms_train, return_size=False)
            trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True)

            valset = tndt_dataset.TNDTDataset(mode='val', fold=fold, transform=composed_transforms_val, return_size=False)
            valloader = DataLoader(valset, batch_size=1, shuffle=False, num_workers=2)

            testset = tndt_dataset.TNDTDataset(mode='test', transform=composed_transforms_test, return_size=False)
            testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)

        # set path
        save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
        save_dir = os.path.join(save_dir_root, 'results', 'model-output', f'run-{args.dataset}', args.backbone + '_lr' + str(args.lr), 'fold' + str(fold))

        models_dir = os.path.join(save_dir, 'models')
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        if os.path.exists(os.path.join(models_dir, 'best_backbone.pth')):
            continue
        
        # set model
        if args.backbone == 'resnet':
            backbone = resnet18(pretrained=True)
            backbone.fc = nn.Linear(in_features=backbone.fc.in_features, out_features=2)
        elif args.backbone == 'densenet':
            backbone = densenet121(pretrained=True)
            backbone.classifier = nn.Linear(in_features=backbone.classifier.in_features, out_features=2)
        elif args.backbone == 'convnext':
            backbone = convnext_tiny(pretrained=True)
            backbone.classifier = nn.Sequential(nn.Flatten(1), nn.Linear(768, 2))
        elif args.backbone == 'vit':
            backbone = vit(pretrained=True)
            backbone.heads = nn.Linear(in_features=backbone.heads.head.in_features, out_features=2)
        elif args.backbone == 'logonet':
            backbone = logonet.logonet18(pretrained=True)
        # elif args.backbone == 'vmamba_unet':
        elif args.backbone == 'swinnet':
            backbone = swinnet(pretrained=True)
        elif args.backbone == 'swinunet':
            backbone = swinunet(pretrained=True)
        elif args.backbone == 'swinunet_refine':
            backbone = swinunet_refine(pretrained=True)
        elif args.backbone == 'swinunet_refine_1':
            backbone = swinunet_refine_1(pretrained=True)
        elif args.backbone == 'swinunet_refine1':
            backbone = swinunet_refine1(pretrained=True)
        elif args.backbone == 'swinunet_refine2':
            backbone = swinunet_refine2(pretrained=True)
        elif args.backbone == 'swinunet_refine3':
            backbone = swinunet_refine3(pretrained=True)
        elif args.backbone == 'swinunet_refine4':
            backbone = swinunet_refine4(pretrained=True)
        elif args.backbone == 'swinunet_refine4_1':
            backbone = swinunet_refine4_1(pretrained=True)
        else:
            raise NotImplementedError
        
        backbone.cuda()
        backbone_optim = optim.SGD(
            backbone.parameters(),
            lr=args.lr,
            momentum=0.9
        )

        # initialize model, optimizer and dataset for training
        if args.resume_epoch == 0:
            print('Training from scratch...')
        else:
            backbone_resume_path = os.path.join(save_dir, 'models', 'backbone_epoch-' + str(args.resume_epoch - 1) + '.pth')
            print('Initializing weights from: {}, epoch: {}...'.format(save_dir, args.resume_epoch))
            backbone.load_state_dict(torch.load(backbone_resume_path, map_location=lambda storage, loc: storage))
        
        # set logs
        log_dir = os.path.join(save_dir, datetime.now().strftime('%b%d_%H-%M-%M%S') + '_' + socket.gethostname())
        writer = SummaryWriter(log_dir=log_dir)

        logger_path = os.path.join(save_dir, 'train_log.txt')
        with open(logger_path, 'w') as f:
            f.write('optim: SGD \nlr=%.4f\nupdate_lr_every=%d\nseed=%d\n' % (args.lr, args.update_lr_every, args.seed))

        num_iter_tr = len(trainloader)
        nitrs = args.resume_epoch * num_iter_tr
        nsamples = args.batch_size * nitrs
        print('each_epoch_num_iter: %d' % (num_iter_tr))

        # curriculum learning settings for training 
        labels_list = []
        recent_losses = []

        start_t = time.time()
        print('Training Network')
        print('save path: '+ save_dir)
        print("model: " + str(args.backbone) + "fold" + str(fold))

        # train!
        best_auc = 0
        for epoch in range(args.resume_epoch, args.nepochs):
            for ii, sample_batched in enumerate(trainloader):
                backbone.train()
                if args.backbone in ['logonet', 'vmamba_unet', 'swinunet', 'swinunet_refine', 'swinunet_refine_1',
                    'swinunet_refine1', 'swinunet_refine2', 'swinunet_refine3', 
                    'swinunet_refine4','swinunet_refine4_1','swinunet_final']:
                    img, label, mask = sample_batched['image'].cuda(), sample_batched['label'].cuda(), sample_batched['mask'].cuda()
                    feats, out_seg = backbone.forward(img)
                    labels_list.append(label)

                    n = out_seg.size(0)
                    loss_class = F.cross_entropy(feats, label)
                    loss_seg = F.binary_cross_entropy(torch.sigmoid(out_seg.view(n, -1)), mask.view(n, -1).float()) + \
                            IOULoss(out_seg, mask)
                    loss = 0.5 * loss_class + loss_seg
                    # loss = 0.5 * loss_class + 1.5 * loss_seg
                    # loss = loss_class + 0.5 * loss_seg
                else:
                    img, label = sample_batched['image'].cuda(), sample_batched['label'].cuda()
                    if args.backbone in ['swinunet_cis']:
                        feats = backbone.forward(img, maskshape_mean, maskshape_std)
                    else:
                        feats = backbone.forward(img)
                    labels_list.append(label)

                    loss = utils.CELoss(logit=feats, target=label, reduction='mean')
                
                backbone_optim.zero_grad()
                loss.backward()
                backbone_optim.step()

                # Get loss
                trainloss = loss.item()
                if len(recent_losses) < args.log_every:
                    recent_losses.append(trainloss)
                else:
                    recent_losses[nitrs % len(recent_losses)] = trainloss

                nitrs += 1
                nsamples += args.batch_size
                if nitrs % args.log_every == 0:
                    meanloss = sum(recent_losses) / len(recent_losses)
                    writer.add_scalar('data/trainloss', meanloss, nsamples)

            print('epoch: %d timecost:%.2f' % (epoch, time.time() - start_t))

            if epoch > 0:
                best_auc = evaluate(backbone, valloader, epoch, fold, best_auc, logger_path, save_dir=save_dir, modelname=args.backbone, mode='val')
                evaluate(backbone, testloader, epoch, fold, best_auc, logger_path, save_dir=save_dir, modelname=args.backbone, mode='test')
            
            if epoch % args.update_lr_every == args.update_lr_every - 1:
                curlr = utils.lr_poly(args.lr, epoch, args.nepochs, 0.9)
                print('(poly lr policy) learning rate: ', curlr)
                backbone_optim = optim.SGD(
                    backbone.parameters(),
                    lr=curlr,
                )

if __name__ == '__main__':
    args = get_arguments()
    main(args)
