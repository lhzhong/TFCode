### 用LeNet网络训练MNIST数据集时的调试记录

经调试参数设置较合理的是：

* 参数初始化函数用truncated_normal，而不是用tf.contrib.layers.xavier_initializer

* 第一个全连阶层个数设置128,不要太大2049

* 千万注意卷积操作中的卷积核大小[1,1,1,1]，不要写成[1,1,1,1,]
