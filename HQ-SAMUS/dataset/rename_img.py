import os
import re
import shutil

data_path = "Dataset_BUSI/Dataset_BUSI_with_GT"
# classNames = ["benign", "malignant", "normal"]
classNames = ["benign", "malignant"]
goal_path = "Breast-BUSI-nnormal"

def num_from_brackets(str1):
    pattern = r"\((.*?)\)"
    return int(re.findall(pattern, str1)[0])

for className in classNames:
    img_path = os.path.join(data_path, className)
    for filename in os.listdir(img_path):
        num = "{:04d}".format(num_from_brackets(filename))
        new_filename = className + num + ".png"
        # print(new_filename)
        if "mask" not in filename:
            shutil.copy(os.path.join(img_path, filename), os.path.join(goal_path, "img", new_filename))
        else:
            shutil.copy(os.path.join(img_path, filename), os.path.join(goal_path, "label", new_filename))


