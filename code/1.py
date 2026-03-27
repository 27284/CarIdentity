import os
import cv2
import numpy as np
import tensorflow as tf

# ===================== 超参数 =====================
IMG_W = 20
IMG_H = 20
CHANNEL = 1
CLASSES = 67
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# ===================== 字符表 =====================
char_table = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
    'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
    'W', 'X', 'Y', 'Z',
    '川', '鄂', '赣', '甘', '贵', '桂', '黑', '沪', '冀',
    '津', '京', '吉', '辽', '鲁', '蒙', '闽', '宁', '青',
    '琼', '陕', '苏', '晋', '皖', '湘', '新', '豫', '渝',
    '粤', '云', '藏', '浙'
]

label2idx = {label: i for i, label in enumerate(char_table)}
idx2label = {i: label for i, label in enumerate(char_table)}

# ===================== 加载数据集 =====================
def load_dataset(root_dir):
    images = []
    labels = []
    for label_name in os.listdir(root_dir):
        folder = os.path.join(root_dir, label_name)
        if not os.path.isdir(folder):
            continue
        if label_name not in label2idx:
            continue
        label = label2idx[label_name]
        for fname in os.listdir(folder):
            path = os.path.join(folder, fname)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (IMG_W, IMG_H))
            img = img / 255.0
            images.append(img)
            labels.append(label)
    images = np.array(images, dtype=np.float32)
    images = np.expand_dims(images, axis=-1)
    labels = np.array(labels, dtype=np.int32)
    return images, labels

# ===================== CNN 模型 =====================
def build_model(x_place, keep_place):
    conv1 = tf.layers.conv2d(x_place, 32, 3, activation=tf.nn.relu, padding='same')
    pool1 = tf.layers.max_pooling2d(conv1, 2, 2, padding='same')

    conv2 = tf.layers.conv2d(pool1, 64, 3, activation=tf.nn.relu, padding='same')
    pool2 = tf.layers.max_pooling2d(conv2, 2, 2, padding='same')

    flat = tf.layers.flatten(pool2)
    fc1 = tf.layers.dense(flat, 256, activation=tf.nn.relu)
    fc1_drop = tf.nn.dropout(fc1, keep_place)

    out = tf.layers.dense(fc1_drop, CLASSES, name='out_put')
    return out

# ===================== 【已修复】训练函数 =====================
def train(train_dir, val_dir, model_save_path):
    os.makedirs(model_save_path, exist_ok=True)

    x_place = tf.placeholder(tf.float32, [None, IMG_H, IMG_W, CHANNEL], name='x_place')
    y_place = tf.placeholder(tf.int32, [None])
    keep_place = tf.placeholder(tf.float32, name='keep_place')

    logits = build_model(x_place, keep_place)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_place))
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

    correct = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), y_place)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    saver = tf.train.Saver(max_to_keep=3)

    # 加载数据
    train_images, train_labels = load_dataset(train_dir)
    val_images, val_labels = load_dataset(val_dir)

    # ======== 修复点1：初始化 val_acc，防止未定义报错 ========
    val_acc = 0.0

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        n_samples = len(train_images)
        step = 0
        max_steps = 2000

        print("开始训练...")
        while step < max_steps:
            indices = np.random.choice(len(train_images), BATCH_SIZE)
            bx = train_images[indices]
            by = train_labels[indices]

            _, train_loss, train_acc = sess.run(
                [optimizer, loss, accuracy],
                feed_dict={x_place: bx, y_place: by, keep_place: 0.5}
            )

            step += 1

            # 每10步验证一次
            if step % 10 == 0:
                val_acc = sess.run(accuracy, feed_dict={
                    x_place: val_images,
                    y_place: val_labels,
                    keep_place: 1.0
                })

                print(f"步 {step:4d} | loss {train_loss:.4f} | train_acc {train_acc:.4f} | val_acc {val_acc:.4f}")

                # ======== 修复点2：判断必须放在里面，val_acc 才有值 ========
                if val_acc > 0.95 and step > 800:
                    print(f"\n🎯 验证精度达标：{val_acc:.4f}，提前保存并停止训练")
                    save_path = saver.save(sess, os.path.join(model_save_path, "model.ckpt"), global_step=step)
                    print("✅ 模型已保存：", save_path)
                    return

        # 跑完最大步也保存
        save_path = saver.save(sess, os.path.join(model_save_path, "model.ckpt"), global_step=step)
        print("\n🏁 训练完成，模型已保存：", save_path)

# ===================== 测试（你要的格式） =====================
def test(test_dir, model_path):
    print("✅ 加载测试集...")
    images, labels = load_dataset(test_dir)
    print(f"📊 测试集共 {len(images)} 张图片")

    print(f"✅ 正在加载模型：{model_path}")

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(f"{model_path}.meta")
        saver.restore(sess, model_path)
        g = tf.get_default_graph()

        x_place = g.get_tensor_by_name("x_place:0")
        keep_place = g.get_tensor_by_name("keep_place:0")
        out_put = g.get_tensor_by_name("out_put/BiasAdd:0")

        pred_idx = tf.argmax(tf.nn.softmax(out_put), axis=1)
        predictions = sess.run(pred_idx, feed_dict={x_place: images, keep_place: 1.0})

    correct = np.sum(predictions == labels)
    acc = correct / len(labels)

    print("=" * 70)
    print(f"🎯 测试集准确率 = {acc:.4f}  ({acc*100:.2f}%)")
    print("=" * 70)

    print("\n📌 测试集每张图片对比结果：")
    for i in range(len(labels)):
        true = idx2label[labels[i]]
        pred = idx2label[predictions[i]]
        print(f"第 {i+1:>2}张 | 真实：{true:<10} | 预测：{pred:<10}")

    print("\n✅ 测试完成！")

# ===================== 主入口 =====================
if __name__ == '__main__':
    TRAIN_DIR   = r"./dataset/train"
    VAL_DIR     = r"./dataset/val"
    TEST_DIR    = r"./dataset/test"
    MODEL_SAVE  = r"./model"
    MODEL_PATH  = r"./model/model.ckpt-810"

    # 训练（跑一次就注释）
    # train(TRAIN_DIR, VAL_DIR, MODEL_SAVE)

    # 测试
    test(TEST_DIR, MODEL_PATH)
