import os

from torch.cuda.amp import GradScaler

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
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
from loss_dzh import DiceLoss, BDLoss
from tqdm import tqdm
from utils.loss_functions.connect_loss import connect_loss
from utils.data_us import JointTransform2D, ImageToImage2D

# 定义参数
def get_parameter():
    parser = argparse.ArgumentParser(description='Networks')
    parser.add_argument('--modelname', default='SAMUS', type=str, help='type of model, e.g., SAM, SAMFull, MedSAM, MSA, SAMed, SAMUS...')
    parser.add_argument('--dir_path', default='dataset/', type=str)
    parser.add_argument('--imgs_path', default='dataset/Breast-BUSI/img', type=str)
    parser.add_argument('--masks_path', default='dataset/Breast-BUSI/label', type=str)
    parser.add_argument('--split_path', default='MainPatient', type=str)
    parser.add_argument('--save_path', default='/data/dzh/SAMUS-HQ/checkpoints/train_boundary/', type=str)
    parser.add_argument('--result_path', default='result/train_boundary/', type=str)
    parser.add_argument('--workers', default=1, type=int)
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--learning_rate', default=1e-4, type=int)
    parser.add_argument('--momentum', default=0.9, type=int)
    parser.add_argument('--classes', default=2, type=int)
    parser.add_argument('--eval_freq', default=10, type=int)
    parser.add_argument('--save_freq', default=30, type=int)
    parser.add_argument('--pre_trained', default=True, type=bool)
    parser.add_argument('--device', default="cuda", type=str)
    parser.add_argument('--input_size', type=int, default=256, help='the image size of the encoder input')
    parser.add_argument('--low_image_size', type=int, default=128, help='the image embedding size')
    parser.add_argument('--vit_name', type=str, default='vit_b', help='select the vit model for the image encoder of sam')
    # parser.add_argument('--sam_ckpt', type=str, default='checkpoints/BUSI_1/SAMUS__260.pth', help='Pretrained checkpoint of SAM')
    parser.add_argument('--sam_ckpt', type=str, default='checkpoints/sam_vit_b_01ec64.pth', help='Pretrained checkpoint of SAM')
    parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
    parser.add_argument('-device', type=str, default='cuda', help='device is cpu or gpu')
    parser.add_argument('--base_lr', type=float, default=0.0005, help='segmentation network learning rate, 0.005 for SAMed, 0.0001 for MSA')  # 0.0006
    parser.add_argument('--warmup', type=bool, default=True, help='If activated, warp up the learning from a lower lr to the base_lr')
    parser.add_argument('--warmup_period', type=int, default=250, help='Warp up iterations, only valid whrn warmup is activated')
    parser.add_argument('--gpu_id', type=str, default='1,3')

    args = parser.parse_args()
    return args


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

def print_num_parameters(model):
    # Find total parameters and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')

def load_from(samus, sam_dict, image_size, patch_size): # load the positional embedding
    samus_dict = samus.state_dict()
    dict_trained = {k: v for k, v in sam_dict.items() if k in samus_dict}
    token_size = int(image_size//patch_size)
    rel_pos_keys = [k for k in dict_trained.keys() if 'rel_pos' in k]
    global_rel_pos_keys = [k for k in rel_pos_keys if '2' in k or '5' in  k or '8' in k or '11' in k]
    for k in global_rel_pos_keys:
        rel_pos_params = dict_trained[k]
        h, w = rel_pos_params.shape
        rel_pos_params = rel_pos_params.unsqueeze(0).unsqueeze(0)
        rel_pos_params = F.interpolate(rel_pos_params, (token_size * 2 - 1, w), mode='bilinear', align_corners=False)
        dict_trained[k] = rel_pos_params[0, 0, ...]
    samus_dict.update(dict_trained)
    return samus_dict

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def dice_coefficient(pred, gt, smooth=1e-5): # output为预测结果 gt 为真实结果
    """ computational formula：
        dice = 2TP/(FP + 2TP + FN)
    """
    pred = torch.sigmoid(pred)
    N = gt.shape[0]
    pred = torch.where(pred > 0.5, 1, 0)
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
    pred = torch.sigmoid(pred)
    N = gt.shape[0]
    pred = torch.where(pred > 0.5, 1, 0)
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


def train_step(args, model, dataloader_train, criterion, optimizer, iter_num, max_iterations):
    model.train()
    total_loss = 0
    for i, (datapack) in tqdm(enumerate(dataloader_train)):
        imgs = datapack['img'].float().to(device=args.device)
        mask = datapack["mask"].float().to(device=args.device)
        pt = get_click_prompt(args, datapack)
        # -------------------------------------------------------- forward --------------------------------------------------------
        output = model(imgs, pt, bbox=None)
        pred = output["masks"].squeeze(1)
        dice_weight = 0.2
        bd_weight = 0.3
        BCE_loss = criterion[0](pred, mask)
        dice_loss = criterion[1](pred, mask)
        boundary_loss = criterion[2](pred, mask)
        train_loss = dice_weight * dice_loss + (1-dice_weight) * BCE_loss + bd_weight * boundary_loss
        total_loss += train_loss.item()
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        # 更新 lr
        if args.warmup and iter_num < args.warmup_period:
            lr_ = args.base_lr * ((iter_num + 1) / args.warmup_period)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
        else:
            if args.warmup:
                shift_iter = iter_num - args.warmup_period
                assert shift_iter >= 0, f'Shift iter is {shift_iter}, smaller than zero'
                lr_ = args.base_lr * (
                        1.0 - shift_iter / max_iterations) ** 0.9  # learning rate adjustment depends on the max iterations
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
        iter_num += 1

    return total_loss / len(dataloader_train), iter_num

def evaluate(args, model, dataloader_val, criterion):
    model.eval()
    total_loss = 0
    total_dice = 0
    total_iou = 0
    for i, (datapack) in tqdm(enumerate(dataloader_val)):
        img = datapack["img"].float().to(device=args.device)
        mask = datapack["mask"].float().to(device=args.device)
        with torch.no_grad():
            pt = get_click_prompt(args, datapack)
            output = model(img, pt, bbox=None)
            pred = output["masks"].squeeze(1)
            dice_weight = 0.2
            bd_weight = 0.3
            BCE_loss = criterion[0](pred, mask)
            dice_loss = criterion[1](pred, mask)
            boundary_loss = criterion[2](pred, mask)
            val_loss = dice_weight * dice_loss + (1 - dice_weight) * BCE_loss + bd_weight * boundary_loss
            total_loss += val_loss.item()
            total_dice += dice_coefficient(pred, mask)
            total_iou += sespiou_coefficient(pred, mask)
    return total_loss / len(dataloader_val), total_dice / len(dataloader_val), total_iou / len(dataloader_val)

def train(args, model, dataloader_train, dataloader_val, epochs):
    best_dice = 0

    W = args.input_size
    H = args.input_size
    hori_translation = torch.zeros([1, 1, W, W])
    for i in range(W - 1):
        hori_translation[:, :, i, i + 1] = torch.tensor(1.0)
    verti_translation = torch.zeros([1, 1, H, H])
    for j in range(H - 1):
        verti_translation[:, :, j, j + 1] = torch.tensor(1.0)
    hori_translation = hori_translation.float()
    verti_translation = verti_translation.float()

    # ***************************** define loss ************************************************
    BCE_Loss = nn.BCEWithLogitsLoss()
    Dice_Loss = DiceLoss()
    Boundary_Loss = BDLoss()
    criterion = [BCE_Loss, Dice_Loss, Boundary_Loss]
    # ***************************** define optimizer *******************************************
    # method 1
    if args.warmup:
        b_lr = args.base_lr / args.warmup_period
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, betas=(0.9, 0.999), weight_decay=0.1)
    else:
        b_lr = args.base_lr
        optimizer = torch.optim.Adam(model.parameters(), lr=b_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    # scaler = GradScaler() # 创建 梯度缩放器，提升性能
    # schedule.step() # 更新 优化器的lr

    # ***************************** train ******************************************************
    iter_num = 0
    max_iterations = epochs * len(dataloader_train)
    for epoch in range(epochs):
        start_time = time.time()
        train_loss, iter_num = train_step(args, model, dataloader_train, criterion, optimizer, iter_num, max_iterations)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.8f}')
        # print('epoch [{}/{}], train loss:{:.4f}'.format(epoch, args.epochs, train_loss))
        if (epoch+1) % args.save_freq == 0 or epoch == args.epochs or (epoch+1 >= 300 and (epoch+1)%10==0):
            if not os.path.isdir(args.save_path):
                os.makedirs(args.save_path)
            save_path = args.save_path + '/epoch_%d_loss_%.4f.pth' % (epoch + 1, train_loss)
            torch.save(model.state_dict(), save_path)
        if (epoch+1) % args.eval_freq == 0:
            val_loss, mean_dice, mean_iou = evaluate(args, model, dataloader_val, criterion)
            print(f'\tVal Loss: {val_loss:.8f}')
            print(f'\tmean_dice: {mean_dice:.8f}')
            print(f'\tmean_iou: {mean_iou:.8f}')
            if mean_dice > best_dice:
                best_dice = mean_dice
                if not os.path.isdir(args.save_path):
                    os.makedirs(args.save_path)
                save_path = args.save_path + '/best_result_%d_dice_%.4f_iou_%.4f.pth' % (epoch + 1, mean_dice, mean_iou)
                torch.save(model.state_dict(), save_path)


def main():
    args = get_parameter()
    seed_everything(42)

    # ***************************** load data **************************************************
    tf_train = JointTransform2D(img_size=args.input_size, low_img_size=args.low_image_size, ori_size=256, crop=None, p_flip=0.0, p_rota=0.5, p_scale=0.5, p_gaussn=0.0,
                                p_contr=0.5, p_gama=0.5, p_distor=0.0, color_jitter_params=None, long_mask=True)  # image reprocessing
    tf_val = JointTransform2D(img_size=args.input_size, low_img_size=args.low_image_size, ori_size=256, crop=None, p_flip=0, color_jitter_params=None, long_mask=True)

    dataset_train = MyDataset(args.dir_path, args.imgs_path, args.masks_path, args.split_path, args.input_size, 'train', tf_train)
    dataset_val = MyDataset(args.dir_path, args.imgs_path, args.masks_path, args.split_path, args.input_size, 'val', tf_val)
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size * args.n_gpu, shuffle=True, num_workers=0, drop_last = True)
    dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size * args.n_gpu, shuffle=False, num_workers=0, drop_last = True)

    # ***************************** define model and load parameter *****************************
    model = get_model_SAMUS(args)
    print_num_parameters(model)
    model.to(args.device)
    if args.pre_trained:
        state_dict = torch.load(args.sam_ckpt)
        new_state_dict = load_from(model, state_dict, 256, 8)
        model.load_state_dict(new_state_dict, strict=False)

    if args.n_gpu > 1:
        model = nn.DataParallel(model)

    train(args, model.float(), dataloader_train, dataloader_val, args.epochs)

if __name__=='__main__':
    main()