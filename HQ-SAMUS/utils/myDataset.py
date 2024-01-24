import json

import cv2
from torch.utils.data import Dataset
import os
from torchvision import transforms
import albumentations as A
from PIL import Image
import numpy as np
from random import randint
import torch
from utils.data_us import JointTransform2D, ImageToImage2D

def random_click(mask, class_id=1):
    indices = np.argwhere(mask == class_id)
    indices[:, [0,1]] = indices[:, [1,0]]
    point_label = 1
    if len(indices) == 0:
        point_label = 0
        indices = np.argwhere(mask != class_id)
        indices[:, [0,1]] = indices[:, [1,0]]
    pt = indices[np.random.randint(len(indices))]
    return pt[np.newaxis, :], [point_label]

def fixed_click(mask, class_id=1):
    indices = np.argwhere(mask == class_id)
    indices[:, [0,1]] = indices[:, [1,0]]
    point_label = 1
    if len(indices) == 0:
        point_label = 0
        indices = np.argwhere(mask != class_id)
        indices[:, [0,1]] = indices[:, [1,0]]
    pt = indices[len(indices)//2]
    return pt[np.newaxis, :], [point_label]

def random_bbox(mask, class_id=1, img_size=256):
    # return box = np.array([x1, y1, x2, y2])
    indices = np.argwhere(mask == class_id) # Y X
    indices[:, [0,1]] = indices[:, [1,0]] # x, y
    if indices.shape[0] ==0:
        return np.array([-1, -1, img_size, img_size])

    minx = np.min(indices[:, 0])
    maxx = np.max(indices[:, 0])
    miny = np.min(indices[:, 1])
    maxy = np.max(indices[:, 1])

    classw_size = maxx-minx+1
    classh_size = maxy-miny+1

    shiftw = randint(int(0.95*classw_size), int(1.05*classw_size))
    shifth = randint(int(0.95*classh_size), int(1.05*classh_size))
    shiftx = randint(-int(0.05*classw_size), int(0.05*classw_size))
    shifty = randint(-int(0.05*classh_size), int(0.05*classh_size))

    new_centerx = (minx + maxx)//2 + shiftx
    new_centery = (miny + maxy)//2 + shifty

    minx = np.max([new_centerx-shiftw//2, 0])
    maxx = np.min([new_centerx+shiftw//2, img_size-1])
    miny = np.max([new_centery-shifth//2, 0])
    maxy = np.min([new_centery+shifth//2, img_size-1])

    return np.array([minx, miny, maxx, maxy])

def fixed_bbox(mask, class_id = 1, img_size=256):
    indices = np.argwhere(mask == class_id) # Y X (0, 1)
    indices[:, [0,1]] = indices[:, [1,0]]
    if indices.shape[0] ==0:
        return np.array([-1, -1, img_size, img_size])
    minx = np.min(indices[:, 0])
    maxx = np.max(indices[:, 0])
    miny = np.min(indices[:, 1])
    maxy = np.max(indices[:, 1])
    return np.array([minx, miny, maxx, maxy])

def correct_dims(*images):
    corr_images = []
    # print(images)
    for img in images:
        if len(img.shape) == 2:
            corr_images.append(np.expand_dims(img, axis=2))
        else:
            corr_images.append(img)
    if len(corr_images) == 1:
        return corr_images[0]
    else:
        return corr_images


def connectivity_matrix(multimask, class_num=1):

    ##### converting segmentation masks to connectivity masks ####

    [_,rows, cols] = multimask.shape
    # batch = 1
    conn = torch.zeros([class_num*8,rows, cols])
    for i in range(class_num):
        mask = multimask[i,:,:]
        # print(mask.shape)
        up = torch.zeros([rows, cols])#move the orignal mask to up
        down = torch.zeros([rows, cols])
        left = torch.zeros([rows, cols])
        right = torch.zeros([rows, cols])
        up_left = torch.zeros([rows, cols])
        up_right = torch.zeros([rows, cols])
        down_left = torch.zeros([rows, cols])
        down_right = torch.zeros([rows, cols])


        up[:rows-1, :] = mask[1:rows,:]
        down[1:rows,:] = mask[0:rows-1,:]
        left[:,:cols-1] = mask[:,1:cols]
        right[:,1:cols] = mask[:,:cols-1]
        up_left[0:rows-1,0:cols-1] = mask[1:rows,1:cols]
        up_right[0:rows-1,1:cols] = mask[1:rows,0:cols-1]
        down_left[1:rows,0:cols-1] = mask[0:rows-1,1:cols]
        down_right[1:rows,1:cols] = mask[0:rows-1,0:cols-1]

        conn[(i*8)+0,:,:] = mask*down_right
        conn[(i*8)+1,:,:] = mask*down
        conn[(i*8)+2,:,:] = mask*down_left
        conn[(i*8)+3,:,:] = mask*right
        conn[(i*8)+4,:,:] = mask*left
        conn[(i*8)+5,:,:] = mask*up_right
        conn[(i*8)+6,:,:] = mask*up
        conn[(i*8)+7,:,:] = mask*up_left

    conn = conn.float()
    conn = conn.squeeze()
    # print(conn.shape)
    return conn

class MyDataset(Dataset):
    def __init__(self, dir_path, imgs_dir, masks_dir, split_path, img_size, split, transform):
        self.dir_path = dir_path
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.split_path = split_path
        self.img_size = img_size
        self.class_dict_file = os.path.join(dir_path, split_path, 'class.json')
        self.split = split


        # self.transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Resize((img_size, img_size))
        # ])
        # self.train_transform = A.Compose([
        #     A.RandomContrast(limit=0.2, p=1),
        #     A.RandomBrightness(limit=0.2, p=1),
        #     A.Rotate(limit=20, p=0.5)
        # ])

        if self.split == 'train':
            with open(os.path.join(dir_path, split_path, "train-Breast-BUSI.txt"), 'r') as f:
                self.data = [line.strip() for line in f]
            f.close()
        elif self.split == 'test':
            with open(os.path.join(dir_path, split_path, "test-Breast-BUSI.txt"), 'r') as f:
                self.data = [line.strip() for line in f]
            f.close()
        else:
            with open(os.path.join(dir_path, split_path, "val-Breast-BUSI.txt"), 'r') as f:
                self.data = [line.strip() for line in f]
            f.close()

        with open(self.class_dict_file, 'r') as load_f:
            self.class_dict = json.load(load_f)

        if transform:
            self.joint_transform = transform
        else:
            to_tensor = transforms.ToTensor()
            self.joint_transform = lambda x, y: (to_tensor(x), to_tensor(y))


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.imgs_dir, self.data[idx].split('/')[-1] + '.png')
        mask_path = os.path.join(self.masks_dir, self.data[idx].split('/')[-1] + '.png')
        # img = self.transform(Image.open(img_path)).unsqueeze(0)
        img = cv2.imread(img_path, 0)
        mask = cv2.imread(mask_path, 0)
        classes = self.class_dict['Breast-BUSI']
        if classes == 2:
            mask[mask > 1] = 1
        # mask = np.where(mask > 0.5, 1, 0)

        image, mask = correct_dims(img, mask)
        if self.joint_transform:
            img, mask, _ = self.joint_transform(img, mask)

        # --------- make the point prompt -----------------
        if 'train' in self.split:
            pt, point_label = random_click(np.array(mask), class_id=1)
            # bbox = random_bbox(np.array(mask), class_id=1, img_size=self.img_size)
        else:
            pt, point_label = fixed_click(np.array(mask), class_id=1)
            # bbox = fixed_bbox(np.array(mask), class_id=1, img_size=self.img_size)
        point_labels = np.array(point_label)

        # image, mask = correct_dims(img, mask)
        # img = self.transform(img)
        # mask = self.transform(mask)
        connectivity_mask = connectivity_matrix(mask.unsqueeze(0), 1)

        # if self.split == 'train':
        #     augment = self.train_transform(image=img, mask=mask)
        #     img, mask = augment['image'], augment['mask']
        return {"img_name": self.data[idx].split('/')[-1] + '.png', "img": img, "mask": mask,
                "connectivity_mask": connectivity_mask, "p_label": point_labels, "pt": pt, "img_shape": img.shape}


    # dataloader 加载 batch_size 的方式，可自定义
    # def collate_fn(self):
    #     return

if __name__=="__main__":
    tf_val = JointTransform2D(img_size=256, low_img_size=128, ori_size=256, crop=None,
                              p_flip=0, color_jitter_params=None, long_mask=True)

    dataset = MyDataset("../dataset/", "../dataset/Breast-BUSI/img", "../dataset/Breast-BUSI/label", 256, 'test', tf_val)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
    datapack = next(iter(dataloader))
    imgs, masks, connectivity_mask = datapack["img"], datapack["mask"], datapack["connectivity_mask"]
    print(torch.max(masks))
    print(torch.sum(connectivity_mask))
    print(imgs.shape, masks.shape, connectivity_mask.shape)
