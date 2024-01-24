1、checkpoints：预训练权重所在文件夹
2、dataset：数据集所在文件夹
    - Breast-BUSI
        - imgs：原图
        - label：标签
    - MainPatient
        - class.json：数据集包含类别
        - train-Breast-BUSI.txt：训练集包含的图片列表
        - test-Breast-BUSI.txt：测试集包含的图片列表
        - val-Breast-BUSI.txt：验证集包含的图片列表
3、model：模型部分代码
4、utils：dataset 定义函数、损失函数等
5、train_hq_boundary.py：模型训练代码
6、get_metric.py：模型评估代码（评价指标：dice、iou）
7、predict_BUSI.py：预测代码

训练步骤：
1、下载预训练权重（SAM 公开的预训练权重）至 checkpoints 文件夹下，并修改 train_hq_boundary.py 中的 sam_ckpt 为相应路径
2、根据以上介绍，将数据集放到 dataset 文件夹下
3、执行 python train_hq_boundary.py 即可
【注】可按需修改其他参数值

预测步骤：
1、修改 predict_BUSI.py 中的 权重加载路径，sam_ckpt
2、执行 python predict_BUSI.py ，图片将保存至 Pred/BUSI（可修改，result_path） 文件夹下

模型评估：
1、修改 get_metric.py 中的 权重加载路径，sam_ckpt
2、执行 python get_metric.py ，评价指标为 dice 系数以及 iou