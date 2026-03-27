import sys
import os
import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split

def augment_plate(img):
    """
    训练数据增强（仅用于 plateNeuralNet 的训练输入）。
    输入 img: (36, 136, 3) uint8
    """
    out = img.copy()

    # 亮度/对比度扰动
    if np.random.rand() < 0.7:
        a = np.random.uniform(0.8, 1.2)   # 对比度
        b = np.random.uniform(-25, 25)    # 亮度偏移
        out = np.clip(out.astype(np.float32) * a + b, 0, 255).astype(np.uint8)

    # 轻微模糊
    if np.random.rand() < 0.3:
        k = 3 if np.random.rand() < 0.7 else 5
        out = cv2.GaussianBlur(out, (k, k), 0)

    # 轻微旋转+平移
    if np.random.rand() < 0.5:
        angle = np.random.uniform(-3, 3)
        dx = np.random.uniform(-2, 2)
        dy = np.random.uniform(-2, 2)
        h, w = out.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
        M[0, 2] += dx
        M[1, 2] += dy
        out = cv2.warpAffine(
            out, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )

    # 少量噪声
    if np.random.rand() < 0.3:
        noise = np.random.normal(0, 8, out.shape).astype(np.float32)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return out

class plate_cnn_net:
    def __init__(self):
        # 输入图片固定为车牌候选框尺寸 136x36，3通道彩色图
        self.img_w,self.img_h = 136,36
        # 二分类：no / has
        self.y_size = 2
        self.batch_size = 100
        self.learn_rate = 0.001

        # x_place: [batch, h, w, c], y_place: one-hot 标签
        self.x_place = tf.placeholder(dtype=tf.float32, shape=[None, self.img_h, self.img_w, 3], name='x_place')
        self.y_place = tf.placeholder(dtype=tf.float32, shape=[None, self.y_size], name='y_place')
        # keep_place 为 dropout 保留率，训练时<1，测试时=1
        self.keep_place = tf.placeholder(dtype=tf.float32, name='keep_place')

    def cnn_construct(self):
        # 输入形状: [-1, 36, 136, 3]
        x_input = tf.reshape(self.x_place, shape=[-1, self.img_h, self.img_w, 3])

        # 卷积块1：卷积提特征 + 池化降采样 + dropout
        cw1 = tf.Variable(tf.random_normal(shape=[3, 3, 3, 32], stddev=0.01), dtype=tf.float32)
        cb1 = tf.Variable(tf.random_normal(shape=[32]), dtype=tf.float32)
        conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x_input, filter=cw1, strides=[1, 1, 1, 1], padding='SAME'), cb1))
        conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv1 = tf.nn.dropout(conv1, self.keep_place)

        # 卷积块2
        cw2 = tf.Variable(tf.random_normal(shape=[3, 3, 32, 64], stddev=0.01), dtype=tf.float32)
        cb2 = tf.Variable(tf.random_normal(shape=[64]), dtype=tf.float32)
        conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, filter=cw2, strides=[1, 1, 1, 1], padding='SAME'), cb2))
        conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv2 = tf.nn.dropout(conv2, self.keep_place)

        # 卷积块3
        cw3 = tf.Variable(tf.random_normal(shape=[3, 3, 64, 128], stddev=0.01), dtype=tf.float32)
        cb3 = tf.Variable(tf.random_normal(shape=[128]), dtype=tf.float32)
        conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, filter=cw3, strides=[1, 1, 1, 1], padding='SAME'), cb3))
        conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv3 = tf.nn.dropout(conv3, self.keep_place)

        # 36x136 经过3次 2x2 池化后，空间尺寸变为 5x17，通道128 -> 展平为 17*5*128
        conv_out = tf.reshape(conv3, shape=[-1, 17 * 5 * 128])

        # 全连接层1
        fw1 = tf.Variable(tf.random_normal(shape=[17 * 5 * 128, 1024], stddev=0.01), dtype=tf.float32)
        fb1 = tf.Variable(tf.random_normal(shape=[1024]), dtype=tf.float32)
        fully1 = tf.nn.relu(tf.add(tf.matmul(conv_out, fw1), fb1))
        fully1 = tf.nn.dropout(fully1, self.keep_place)

        # 全连接层2
        fw2 = tf.Variable(tf.random_normal(shape=[1024, 1024], stddev=0.01), dtype=tf.float32)
        fb2 = tf.Variable(tf.random_normal(shape=[1024]), dtype=tf.float32)
        fully2 = tf.nn.relu(tf.add(tf.matmul(fully1, fw2), fb2))
        fully2 = tf.nn.dropout(fully2, self.keep_place)

        # 输出层（logits），名称 out_put 供其他脚本恢复图时按名字取张量
        fw3 = tf.Variable(tf.random_normal(shape=[1024, self.y_size], stddev=0.01), dtype=tf.float32)
        fb3 = tf.Variable(tf.random_normal(shape=[self.y_size]), dtype=tf.float32)
        fully3 = tf.add(tf.matmul(fully2, fw3), fb3, name='out_put')

        return fully3

    def train(self,data_dir,model_save_path):
        print('ready load train dataset')
        # 从目录读取样本和 one-hot 标签
        X, y = self.init_data(data_dir)
        print('success load ' + str(len(y)) + ' datas')
        # 划分训练集/测试集
        train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=0)

        out_put = self.cnn_construct()
        # 预测类别/真实类别用于计算准确率
        predicts = tf.nn.softmax(out_put)
        predicts = tf.argmax(predicts, axis=1)
        actual_y = tf.argmax(self.y_place, axis=1)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(predicts, actual_y), dtype=tf.float32))
        # 训练目标：交叉熵最小化
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out_put, labels=self.y_place))
        opt = tf.train.AdamOptimizer(self.learn_rate)
        train_step = opt.minimize(cost)

        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            step = 0
            saver = tf.train.Saver()
            while True:
                # 随机采样一个 batch 训练
                train_index = np.random.choice(len(train_x), self.batch_size, replace=False)
                train_randx = train_x[train_index]
                train_randy = train_y[train_index]
                # 只增强训练 batch：不增强测试集/评估集
                train_randx_aug = np.stack([augment_plate(img) for img in train_randx], axis=0).astype(np.float32)
                _, loss = sess.run([train_step, cost], feed_dict={self.x_place: train_randx_aug,
                                                                  self.y_place: train_randy, self.keep_place: 0.75})
                step += 1
                print(step, loss)

                if step % 10 == 0:
                    # 每10步做一次抽样评估
                    test_index = np.random.choice(len(test_x), self.batch_size, replace=False)
                    test_randx = test_x[test_index]
                    test_randy = test_y[test_index]
                    # 测试集不做增强（评估更公平）
                    acc = sess.run(accuracy, feed_dict={self.x_place: test_randx,
                                                        self.y_place: test_randy, self.keep_place: 1.0})
                    print('accuracy:' + str(acc))
                    # 达到目标精度后保存并结束训练
                    if (acc > 0.99 and step > 500) or step > 1000 :
                        saver.save(sess, model_save_path, global_step=step)
                        break

    def test(self,x_images,model_path):
        # 与训练时同结构，加载参数后前向推理
        out_put = self.cnn_construct()
        predicts = tf.nn.softmax(out_put)
        probabilitys = tf.reduce_max(predicts, reduction_indices=[1])
        predicts = tf.argmax(predicts, axis=1)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, model_path)
            preds, probs = sess.run([predicts, probabilitys], feed_dict={self.x_place: x_images, self.keep_place: 1.0})
        return preds,probs

    def list_all_files(self,root):
        # 递归列出目录下所有文件
        files = []
        list = os.listdir(root)
        for i in range(len(list)):
            element = os.path.join(root, list[i])
            if os.path.isdir(element):
                files.extend(self.list_all_files(element))
            elif os.path.isfile(element):
                files.append(element)
        return files

    def init_data(self,dir):
        # 训练数据目录结构示例：
        # cnn_plate_train/has/*.jpg
        # cnn_plate_train/no/*.jpg
        X = []
        y = []
        if not os.path.exists(dir):
            raise ValueError('没有找到文件夹')
        files = self.list_all_files(dir)
        labels = [os.path.split(os.path.dirname(file))[-1] for file in files]

        for i, file in enumerate(files):
            src_img = cv2.imread(file)
            if src_img.ndim != 3:
                continue
            # 统一输入尺寸
            resize_img = cv2.resize(src_img, (136, 36))
            X.append(resize_img)
            # 标签约定：has -> [0,1]，其余（如 no）-> [1,0]
            y.append([[0, 1] if labels[i] == 'has' else [1, 0]])

        X = np.array(X)
        y = np.array(y).reshape(-1, 2)
        return X, y

    def init_testData(self,dir):
        # 测试数据读取：返回 (test_X, test_y)
        # 目录结构要求：
        #   cnn_plate_test/has/*.jpg
        #   cnn_plate_test/no/*.jpg
        test_X = []
        test_y = []
        if not os.path.exists(dir):
            raise ValueError('没有找到文件夹')
        files = self.list_all_files(dir)
        labels = [os.path.split(os.path.dirname(file))[-1] for file in files]

        for i, file in enumerate(files):
            src_img = cv2.imread(file)
            if src_img is None or src_img.ndim != 3:
                continue
            resize_img = cv2.resize(src_img, (136, 36))
            test_X.append(resize_img)
            # 标签约定：has -> [0,1]；no/其他 -> [1,0]
            test_y.append([[0, 1] if labels[i] == 'has' else [1, 0]])

        test_X = np.array(test_X)
        test_y = np.array(test_y).reshape(-1, 2)
        return test_X, test_y


if __name__ == '__main__':
    cur_dir = sys.path[0]
    project_dir = os.path.abspath(os.path.join(cur_dir, '..'))
    # 训练集、测试集、模型保存路径
    data_dir = os.path.join(project_dir, 'carIdentityData', 'cnn_plate_train')
    test_dir = os.path.join(project_dir, 'carIdentityData', 'cnn_plate_test')
    train_model_path = os.path.join(project_dir, 'carIdentityData', 'model', 'plate_recongnize', 'model.ckpt')
    model_path = os.path.join(project_dir,'carIdentityData', 'model', 'plate_recongnize', 'model.ckpt-510')

    # 1=训练，0=测试
    train_flag = 0
    net = plate_cnn_net()

    if train_flag == 1:
        # 训练模型
        model_dir = os.path.dirname(train_model_path)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        net.train(data_dir,train_model_path)
    else:
        # 测试部分
        test_X, test_y = net.init_testData(test_dir)
        preds,probs = net.test(test_X,model_path)
        # 计算测试集 accuracy（与训练计算方式一致）
        actual_y = np.argmax(test_y, axis=1)
        acc = float(np.mean(preds == actual_y))
        print('test accuracy:', acc)
        for i in range(len(preds)):
            pred = preds[i].astype(int)
            prob = probs[i]
            if pred == 1:
                print('plate',prob)
            else:
                print('no',prob)
