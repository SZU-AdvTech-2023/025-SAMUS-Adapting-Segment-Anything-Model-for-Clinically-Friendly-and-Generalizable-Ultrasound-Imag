import os
import random

random.seed(1234)

path = "Breast-BUSI/img"
ratio = [0.7, 0.2, 0.1]
img_list = []

for filename in os.listdir(path):
    img_list.append(filename.split('.')[0])

random.shuffle(img_list)

train_list = img_list[:int(len(img_list) * ratio[0])]
val_list = img_list[int(len(img_list) * ratio[0]):int(len(img_list) * (ratio[0]+ratio[1]))]
test_list = img_list[int(len(img_list) * (ratio[0]+ratio[1])):]

with open("train-Breast-BUSI.txt", "w") as file:
    for img_name in train_list:
        if "normal" in img_name:
            file.write("0/Breast-BUSI/" + img_name + '\n')
        else:
            file.write("1/Breast-BUSI/" + img_name + '\n')

with open("test-Breast-BUSI.txt", "w") as file:
    for img_name in test_list:
        file.write("Breast-BUSI/" + img_name + '\n')

with open("val-Breast-BUSI.txt", "w") as file:
    for img_name in val_list:
        if "normal" in img_name:
            file.write("0/Breast-BUSI/" + img_name + '\n')
        else:
            file.write("1/Breast-BUSI/" + img_name + '\n')