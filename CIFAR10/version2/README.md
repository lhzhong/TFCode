## code改进说明

该程序实现的是CIFAR10的训练,测试及微调

version2中是对原代码的改进，各函数作用如下：

* ```models```文件夹下放了三个模型文件(LeNet,AlexNet,VGG16),其中AlexNet和VGG16里也有加载微调模型的函数
* ```pretrain```文件夹下存放的是在网上下载的已经训练过得模型文件
* ```cafar10_input```是数据读取函数，下载的cifar10的是二进制形式，该函数通过读取二进制形式数据并将之形成batch
* ```cafar10_train```是数据训练函数
* ```cafar10_finetune```是数据微调函数
* ```cafar10_evaluate```是数据测试函数
* ```tools```存放几个其他要调用的函数

