## Dataset API

这是tensorflow读取数据的一种新的方式，之前用过tfrecord方式读取数据，但是tfrecord是把数据生成tfrecord格式存储在内存中，然后再在内存中按队列方式读取数据，存在的问题在于：

* 需要生成tfrecord格式存储，无法实现随取随用的方式
* 生成的tfrecord比原始数据还大，而且需要存储在硬盘中，对于数据量大的对于硬盘资源存放也是一种浪费
* 因为tfrecord存储时是以二进制形式存储，我读取tfrecord内容显示原始图片在生成tfrecord再读取过后会有图片失真现象

Dataset API是tensorflow一种新的数据存储方式，其是TensorFlow 1.3版本中引入的一个新的模块，主要服务于数据读取，构建输入数据的pipeline。

在TensorFlow 1.3中，Dataset API是放在contrib包中的：

> tf.contrib.data.Dataset

而在TensorFlow 1.4中，Dataset API已经从contrib包中移除，变成了核心API的一员：

> tf.data.Dataset

在该文件中，我主要以Alexnet模型为主，将Alexnet写成独立的类，类下同时包含了finetune所需的函数，因为我在fintune脚本里写的是用在IMAGENET上训练的AlexNet来进行微调自己的模型。

程序运行顺序：
1. 运行```data_txt.py```脚本生成源数据对应的txt文件，txt文件里包含了图片路径及标签
2. 运行```data_mean.py```脚本生成训练数据的平均值，并将生成的值生成保存为mean.pkl文件，因为作为预处理阶段数据需要进行去中心处理
3. 运行```train.py```或```finetune.py```脚本，重新训练或微调模型
4. 运行```test.py```脚本， 对训练模型进行测试
