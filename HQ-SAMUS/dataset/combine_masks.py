# 合并同一张 image 的多个masks
import os
import re
import cv2
import shutil

data_path = "Dataset_BUSI/Dataset_BUSI_with_GT"
classNames = ["benign", "malignant", "normal"]

def num_from_brackets(str1):
    pattern = r"\((.*?)\)"
    return int(re.findall(pattern, str1)[0])

for className in classNames:
    img_path = os.path.join(data_path, className)
    rep_mask_imgname = []
    for filename in os.listdir(img_path):
        if "mask_" in filename:
            rep_mask_imgname.append(filename)
    for mask_name in rep_mask_imgname:
        img1 = cv2.imread(os.path.join(img_path, mask_name.split('.')[0][:-2] + ".png"))
        img2 = cv2.imread(os.path.join(img_path, mask_name))
        # print(img2[0, 0])
        for i in range(img2.shape[0]):
            for j in range(img2.shape[1]):
                if img2[i, j, 0] == 255:
                    img1[i, j] = [255, 255, 255]
        os.remove(os.path.join(img_path, mask_name))
        os.remove(os.path.join(img_path, mask_name.split('.')[0][:-2] + ".png"))
        cv2.imwrite(os.path.join(img_path, mask_name.split('.')[0][:-2] + ".png"), img1)