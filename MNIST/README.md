### MNIST数据集调试

该程序实现的是MNIST的训练,测试,显示混淆矩阵等实验，实验测试了两个模型```lenet-300-100```和```lenet5```

**MNIST介绍：**

MNIST由6万张训练图片和1万张测试图片构成的，每张图片都是28*28大小,而且都是黑白色构成

**各函数作用：**

```lenet_300_100_train.py```和```lenet5_train.py```分别训练```lenet-300-100```和```lenet5```模型

```test_model.py```用来测试模型

```models.py```和```tools.py```分别存放模型和一些其他函数

```show_raw_image.py```显示手写体原始图片

```show_confusion_matrix.py```用来显示测试的混淆矩阵

```test_one_image.py```用来测试我们自己的数字图片，但可能由于预处理效果不佳，最后的识别效果也不太好

**识别率:**

| Model                 | Accuracy   | 
| -------------         |:----------:| 
| LeNet-300-100         | 98.15%     | 
| LeNet5                | 99.23%     |  


