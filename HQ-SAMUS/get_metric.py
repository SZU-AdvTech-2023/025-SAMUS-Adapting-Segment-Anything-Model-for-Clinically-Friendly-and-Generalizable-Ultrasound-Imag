import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
import argparse
import torch
import numpy as np
import random
import torchvision
from torch import nn, optim
from torch.autograd import Variable
from models.model_dict import get_model_SAMUS
from utils.myDataset import MyDataset
from torch.utils.data import DataLoader
import time
from utils.loss_functions.sam_loss import Mask_DC_and_BCE_loss
from torch.nn import functional as F
from utils.generate_prompts import get_click_prompt
import cv2
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from utils.data_us import JointTransform2D, ImageToImage2D

# 定义参数
def get_parameter():
    parser = argparse.ArgumentParser(description='Networks')
    parser.add_argument('--modelname', default='SAMUS', type=str, help='type of model, e.g., SAM, SAMFull, MedSAM, MSA, SAMed, SAMUS...')
    parser.add_argument('--dir_path', default='dataset/', type=str)
    parser.add_argument('--imgs_path', default='dataset/Breast-BUSI/img', type=str)
    parser.add_argument('--masks_path', default='dataset/Breast-BUSI/label', type=str)
    parser.add_argument('--split_path', default='MainPatient', type=str)
    parser.add_argument('--result_path', default='result/BUSI/', type=str)
    parser.add_argument('--workers', default=1, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--classes', default=2, type=int)
    parser.add_argument('--pre_trained', default=True, type=bool)
    parser.add_argument('--device', default="cuda", type=str)
    parser.add_argument('--input_size', type=int, default=256, help='the image size of the encoder input')
    parser.add_argument('--encoder_input_size', type=int, default=256, help='the image size of the encoder input')
    parser.add_argument('--low_image_size', type=int, default=128, help='the image embedding size')
    parser.add_argument('--vit_name', type=str, default='vit_b', help='select the vit model for the image encoder of sam')
    parser.add_argument('--sam_ckpt', type=str, default='/data/dzh/SAMUS-HQ/checkpoints/train_boundary/best_result_290_dice_0.8970_iou_0.8304.pth', help='Pretrained checkpoint of SAM')
    parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
    args = parser.parse_args()
    return args

def dice_coefficient(pred, gt, smooth=1e-5): # output为预测结果 gt 为真实结果
    """ computational formula：
        dice = 2TP/(FP + 2TP + FN)
    """
    # pred = torch.sigmoid(pred)
    # pred = torch.where(pred > 0.5, 1, 0)
    N = gt.shape[0]
    pred_flat = pred.reshape(N, -1)
    gt_flat = gt.reshape(N, -1)
    # if (pred.sum() + gt.sum()) == 0:
    #     return 1
    intersection = (pred_flat * gt_flat).sum(1)
    unionset = pred_flat.sum(1) + gt_flat.sum(1)
    dice = (2 * intersection + smooth) / (unionset + smooth)
    return dice.sum() / N


def sespiou_coefficient(pred, gt, smooth=1e-5):
    """ computational formula:
        iou = TP/(FP+TP+FN)
    """
    # pred = torch.sigmoid(pred)
    # pred = torch.where(pred > 0.5, 1, 0)
    N = gt.shape[0]
    pred_flat = pred.reshape(N, -1)
    gt_flat = gt.reshape(N, -1)
    TP = (pred_flat * gt_flat).sum(1)
    FN = gt_flat.sum(1) - TP
    pred_flat_no = (pred_flat + 1) % 2
    gt_flat_no = (gt_flat + 1) % 2
    TN = (pred_flat_no * gt_flat_no).sum(1)
    FP = pred_flat.sum(1) - TP
    IOU = (TP + smooth) / (FP + TP + FN + smooth)
    return IOU.sum() / N


def seed_everything(seed):
    if seed >= 10000:
        raise ValueError("seed number should be less than 10000")
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0
    seed = (rank * 100000) + seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_prompt(args, datapack):
    pt = datapack['pt']
    point_labels = datapack['p_label']
    point_coords = pt
    coords_torch = torch.as_tensor(point_coords, dtype=torch.float32, device=args.device)
    labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=args.device)
    if len(pt.shape) == 2:
        coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
    pt = (coords_torch, labels_torch)
    return pt

def main():
    args = get_parameter()
    seed_everything(42)
    # ***************************** load data **************************************************
    tf_val = JointTransform2D(img_size=args.input_size, low_img_size=args.low_image_size, ori_size=256, crop=None, p_flip=0, color_jitter_params=None, long_mask=True)
    dataset_test = MyDataset(args.dir_path, args.imgs_path, args.masks_path, args.split_path, args.input_size, 'test', tf_val)
    dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=True, num_workers=0)
    # ***************************** define model and load parameter *****************************
    model = get_model_SAMUS(args).to(args.device)

    state_dict = torch.load(args.sam_ckpt)
    model.load_state_dict(state_dict)
    model.eval()
    dice_total = 0
    iou_total = 0
    with torch.no_grad():
        for i, (datapack) in enumerate(dataloader_test):
            imgs = datapack['img'].float().to(device=args.device)
            masks = datapack['mask']
            pt = get_click_prompt(args, datapack)
            pred = model(imgs, pt, bbox=None)['masks'].squeeze(1).cpu()

            pred = torch.sigmoid(pred)
            pred = torch.where(pred >= 0.5, 1, 0)

            # print(datapack['img_name'])
            # print(dice_coefficient(pred, masks).item())
            # print(sespiou_coefficient(pred, masks).item())

            dice_total += dice_coefficient(pred, masks).item()
            iou_total += sespiou_coefficient(pred, masks).item()
        #     count += 1
        # print(count)
        # print(len(dataloader_test))
        # print('dice: ' + str(dice_total / count))
        # print('iou: ' + str(iou_total / count))
        print('dice: ' + str(dice_total / len(dataloader_test)))
        print('iou: ' + str(iou_total / len(dataloader_test)))




if __name__=='__main__':
    main()


