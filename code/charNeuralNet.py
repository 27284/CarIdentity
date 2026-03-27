import sys
import os
import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split

# -*- coding: utf-8 -*-

def augment_char(img):
    out = img.copy()
    if np.random.rand() < 0.7:
        a = np.random.uniform(0.9, 1.1)
        b = np.random.uniform(-10, 10)
        out = np.clip(out.astype(np.float32) * a + b, 0, 255).astype(np.uint8)

    if np.random.rand() < 0.7:
        angle = np.random.uniform(-2, 2)
        dx = np.random.uniform(-1.5, 1.5)
        dy = np.random.uniform(-1.5, 1.5)
        h, w = out.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
        M[0, 2] += dx
        M[1, 2] += dy
        out = cv2.warpAffine(
            out, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )

    if np.random.rand() < 0.3:
        out = cv2.GaussianBlur(out, (3, 3), 0)

    if np.random.rand() < 0.2:
        noise = np.random.normal(0, 5, out.shape).astype(np.float32)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return out

numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
             'U', 'V', 'W', 'X', 'Y', 'Z']
chinese = ['zh_cuan', 'zh_e', 'zh_gan', 'zh_gan1', 'zh_gui', 'zh_gui1', 'zh_hei', 'zh_hu', 'zh_ji', 'zh_jin',
           'zh_jing', 'zh_jl', 'zh_liao', 'zh_lu', 'zh_meng', 'zh_min', 'zh_ning', 'zh_qing', 'zh_qiong',
           'zh_shan', 'zh_su', 'zh_sx', 'zh_wan', 'zh_xiang', 'zh_xin', 'zh_yu', 'zh_yu1', 'zh_yue', 'zh_yun',
           'zh_zang', 'zh_zhe']

class char_cnn_net:
    def __init__(self):
        self.dataset = numbers + alphabets + chinese
        self.dataset_len = len(self.dataset)
        self.img_size = 20
        self.y_size = len(self.dataset)
        self.batch_size = 100

        self.x_place = tf.placeholder(dtype=tf.float32, shape=[None, self.img_size, self.img_size], name='x_place')
        self.y_place = tf.placeholder(dtype=tf.float32, shape=[None, self.y_size], name='y_place')
        self.keep_place = tf.placeholder(dtype=tf.float32, name='keep_place')

    def cnn_construct(self):
        x_input = tf.reshape(self.x_place, shape=[-1, 20, 20, 1])

        cw1 = tf.Variable(tf.random_normal(shape=[3, 3, 1, 32], stddev=0.01), dtype=tf.float32)
        cb1 = tf.Variable(tf.random_normal(shape=[32]), dtype=tf.float32)
        conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x_input, filter=cw1, strides=[1, 1, 1, 1], padding='SAME'), cb1))
        conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv1 = tf.nn.dropout(conv1, self.keep_place)

        cw2 = tf.Variable(tf.random_normal(shape=[3, 3, 32, 64], stddev=0.01), dtype=tf.float32)
        cb2 = tf.Variable(tf.random_normal(shape=[64]), dtype=tf.float32)
        conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, filter=cw2, strides=[1, 1, 1, 1], padding='SAME'), cb2))
        conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv2 = tf.nn.dropout(conv2, self.keep_place)

        cw3 = tf.Variable(tf.random_normal(shape=[3, 3, 64, 128], stddev=0.01), dtype=tf.float32)
        cb3 = tf.Variable(tf.random_normal(shape=[128]), dtype=tf.float32)
        conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, filter=cw3, strides=[1, 1, 1, 1], padding='SAME'), cb3))
        conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv3 = tf.nn.dropout(conv3, self.keep_place)

        conv_out = tf.reshape(conv3, shape=[-1, 3 * 3 * 128])

        fw1 = tf.Variable(tf.random_normal(shape=[3 * 3 * 128, 1024], stddev=0.01), dtype=tf.float32)
        fb1 = tf.Variable(tf.random_normal(shape=[1024]), dtype=tf.float32)
        fully1 = tf.nn.relu(tf.add(tf.matmul(conv_out, fw1), fb1))
        fully1 = tf.nn.dropout(fully1, self.keep_place)

        fw2 = tf.Variable(tf.random_normal(shape=[1024, 1024], stddev=0.01), dtype=tf.float32)
        fb2 = tf.Variable(tf.random_normal(shape=[1024]), dtype=tf.float32)
        fully2 = tf.nn.relu(tf.add(tf.matmul(fully1, fw2), fb2))
        fully2 = tf.nn.dropout(fully2, self.keep_place)

        fw3 = tf.Variable(tf.random_normal(shape=[1024, self.dataset_len], stddev=0.01), dtype=tf.float32)
        fb3 = tf.Variable(tf.random_normal(shape=[self.dataset_len]), dtype=tf.float32)
        fully3 = tf.add(tf.matmul(fully2, fw3), fb3, name='out_put')

        return fully3

    def train(self, data_dir, save_model_path):
        print('✅ 准备加载训练数据集...')
        X, y = self.init_data(data_dir)
        print(f'✅ 成功加载 {len(y)} 条数据')

        train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=0)

        out_put = self.cnn_construct()
        predicts = tf.nn.softmax(out_put)
        predicts = tf.argmax(predicts, axis=1)
        actual_y = tf.argmax(self.y_place, axis=1)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(predicts, actual_y), dtype=tf.float32))
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out_put, labels=self.y_place))
        opt = tf.train.AdamOptimizer(learning_rate=0.001)
        train_step = opt.minimize(cost)

        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            step = 0
            saver = tf.train.Saver(max_to_keep=5)
            val_acc = 0.0

            while True:
                train_index = np.random.choice(len(train_x), self.batch_size, replace=False)
                train_randx = train_x[train_index]
                train_randy = train_y[train_index]
                train_randx_aug = np.stack([augment_char(img) for img in train_randx], axis=0).astype(np.float32)

                _, loss = sess.run([train_step, cost],
                                   feed_dict={self.x_place: train_randx_aug, self.y_place: train_randy, self.keep_place: 0.75})
                step += 1

                if step % 10 == 0:
                    test_index = np.random.choice(len(test_x), self.batch_size, replace=False)
                    test_randx = test_x[test_index]
                    test_randy = test_y[test_index]
                    val_acc = sess.run(accuracy, feed_dict={
                        self.x_place: test_randx,
                        self.y_place: test_randy,
                        self.keep_place: 1.0
                    })
                    print(f"步骤 {step:4d} | loss: {loss:.4f} | accuracy: {val_acc:.4f}")



                    if (val_acc > 0.95 and step > 500) or step >=1000:
                        saver.save(sess, save_model_path, global_step=step)
                        print(f"🎯 训练完成！准确率: {val_acc:.4f}")
                        break

    def test_with_accuracy(self, test_dir, model_path):
        print("✅ 加载测试集...")
        test_X, test_y = self.init_data(test_dir)
        print(f"📊 测试集共 {len(test_X)} 张图片")
        print(f"✅ 正在加载模型：{model_path}")

        out_put = self.cnn_construct()
        predicts = tf.argmax(tf.nn.softmax(out_put), axis=1)
        actual_y = tf.argmax(self.y_place, axis=1)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(predicts, actual_y), dtype=tf.float32))

        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, model_path)
            acc = sess.run(accuracy, feed_dict={self.x_place: test_X, self.y_place: test_y, self.keep_place: 1.0})
            preds = sess.run(predicts, feed_dict={self.x_place: test_X, self.keep_place: 1.0})
            actuals = sess.run(actual_y, feed_dict={self.x_place: test_X, self.y_place: test_y, self.keep_place: 1.0})

        print("=" * 70)
        print(f"🎯 测试集准确率 = {acc:.4f}  ({acc * 100:.2f}%)")
        print("=" * 70)

        print("\n📌 测试集每张图片对比结果：")
        for i in range(len(actuals)):
            true_str = self.dataset[actuals[i]]
            pred_str = self.dataset[preds[i]]
            print(f"第 {i + 1:>3}张 | 真实：{true_str:<12} | 预测：{pred_str:<12}")

        print("\n✅ 测试完成！")
        return acc

    def list_all_files(self, root):
        files = []
        if not os.path.exists(root):
            return files
        list_dir = os.listdir(root)
        for item in list_dir:
            path = os.path.join(root, item)
            if os.path.isdir(path):
                if item in self.dataset:
                    files.extend(self.list_all_files(path))
            elif os.path.isfile(path):
                files.append(path)
        return files

    def init_data(self, dir):
        X = []
        y = []
        if not os.path.exists(dir):
            raise ValueError(f'❌ 没有找到文件夹: {dir}')
        files = self.list_all_files(dir)

        for file in files:
            src_img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            if src_img is None or src_img.ndim != 2:
                continue
            resize_img = cv2.resize(src_img, (20, 20))
            X.append(resize_img)

            dir_name = os.path.basename(os.path.dirname(file))
            vector_y = [0] * self.dataset_len
            vector_y[self.dataset.index(dir_name)] = 1
            y.append(vector_y)

        return np.array(X), np.array(y)

if __name__ == '__main__':
    # 从 code 目录 返回上一级 -> 访问 carIdentityData
    data_dir = '../carIdentityData/cnn_char_train/train'
    train_model_path = '../carIdentityData/model/char_recongnize/model.ckpt'
    model_path = '../carIdentityData/model/char_recongnize/model.ckpt-810'

    train_flag = 0
    net = char_cnn_net()

    if train_flag == 1:
        net.train(data_dir, train_model_path)
    else:
        net.test_with_accuracy(data_dir, model_path)