## code改进说明

该程序实现的是CIFAR10的训练,测试及微调
> version2中是对原代码的改进

**CIFAR10介绍：**

CIFAR10由60000张32*32的 RGB 彩色图片构成，共10个分类。50000张训练，10000张测试（交叉验证）

**各函数作用:**

* ```models```文件夹下放了三个模型文件(LeNet,AlexNet,VGG16),其中AlexNet和VGG16里也有加载微调模型的函数
* ```pretrain```文件夹下存放的是在网上下载的已经训练过得模型文件
* ```cafar10_input```是数据读取函数，下载的cifar10的是二进制形式，该函数通过读取二进制形式数据并将之形成batch
* ```cafar10_train```是数据训练函数
* ```cafar10_finetune```是数据微调函数
* ```cafar10_evaluate```是数据测试函数
* ```tools```存放几个其他要调用的函数

**识别率:**

| Model                 | Accuracy   | 
| -------------         |:----------:| 
| LeNet5                | 76.10%     | 
| AlexNet               | 65.11%     |  
| AlexNet with finetune | 70.43%     |

我的实验结果是虽然Alexnet网络结构比Lenet5深，但是可能由于数据较少，深得网络不一定能取得较好的识别效果
