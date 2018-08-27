import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def get_batch(file, image_h, image_w, batch_size, capacity=256):

    img_label = []
    with open(file, 'r') as file_to_read:
        for line in file_to_read.readlines():
            line = line.strip('\n')
            tmp = line.split(" ")
            img_label.append(tmp)
        img_label = np.array(img_label)
        image = img_label[:, 0]
        label = img_label[:, -1]
        label = np.int32(label)

    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int64)

    # 创建文件名队列
    input_queue = tf.train.slice_input_producer([image, label], shuffle=False)

    # 从文件名队列读取数据到内存队列中
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    ######################################
    # data argumentation should go to here
    ######################################

    image = tf.image.resize_image_with_crop_or_pad(image, image_h, image_w)
    # image = tf.image.resize_images(image, [image_h, image_w], method=0)
    # image = tf.image.per_image_standardization(image)

    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=batch_size,
                                              num_threads=2,  # 设置线程数，设置大了读的数据乱了，我也不知道为什么.....
                                              capacity=capacity)  # 队列里存放的最大样本数

    label_batch = tf.reshape(label_batch, [batch_size])
    # image_batch = tf.cast(image_batch, tf.float32)

    return image_batch, label_batch


if __name__ == '__main__':

    img_batch, lab_batch = get_batch(file='datalist.txt',
                                     image_h=224,
                                     image_w=224,
                                     batch_size=40)

    with tf.Session() as sess:

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            while not coord.should_stop():
                imgs, labs = sess.run([img_batch, lab_batch])
                plt.figure(figsize=(10, 10))
                for i in np.arange(40):
                    plt.subplot(5, 8, i + 1)
                    plt.axis('off')
                    plt.imshow(imgs[i, :, :, :])
                    plt.title(labs[i])
                plt.show()
                break
        except tf.errors.OutOfRangeError:
            print("done")
        finally:
            coord.request_stop()
        coord.join(threads)
