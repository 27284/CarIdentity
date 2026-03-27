# -*- coding: utf-8 -*-
import os
import shutil
import random
from pathlib import Path

# ====================== 【已适配你当前的路径，直接用】 ======================
# 你的 cnn_char_train 根目录（完全对应你截图里的路径）
ROOT = Path(r"D:\Github\CarPlateIdentity-master\carIdentityData\cnn_char_train")
# ==========================================================================

# 严格按 7:2:1 比例划分（train:val:test）
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

# 固定随机种子，保证每次划分结果一致，可复现
random.seed(42)


def split_dataset_move():
    # 源目录：train 文件夹（里面有 0、1、A、B 等类别子文件夹）
    src_train_dir = ROOT / "train"
    # 目标目录：val、test（自动创建对应类别文件夹）
    target_val_dir = ROOT / "val"
    target_test_dir = ROOT / "test"

    # 1. 检查源目录是否存在
    if not src_train_dir.exists():
        raise FileNotFoundError(f"❌ 源 train 目录不存在，请检查路径：{src_train_dir}")

    # 2. 获取所有类别（train 下的子文件夹名）
    classes = [d for d in os.listdir(src_train_dir) if os.path.isdir(src_train_dir / d)]
    print(f"✅ 找到 {len(classes)} 个字符类别")

    # 3. 自动为 val、test 创建和 train 完全一致的类别子文件夹
    for cls in classes:
        # 创建 val 下的类别文件夹
        val_cls_dir = target_val_dir / cls
        val_cls_dir.mkdir(parents=True, exist_ok=True)
        # 创建 test 下的类别文件夹
        test_cls_dir = target_test_dir / cls
        test_cls_dir.mkdir(parents=True, exist_ok=True)
    print("✅ 已为 val/test 自动创建所有类别子文件夹")

    # 4. 按类别逐个划分，剪切移动图片（保证无交集）
    total_val = 0
    total_test = 0

    for cls in classes:
        cls_src_path = src_train_dir / cls
        # 获取该类别下所有图片文件
        img_files = [f for f in os.listdir(cls_src_path) if os.path.isfile(cls_src_path / f)]

        if not img_files:
            print(f"⚠️ 类别 {cls} 下无图片，跳过")
            continue

        # 打乱图片顺序，保证随机划分
        random.shuffle(img_files)
        n_total = len(img_files)

        # 计算各集合数量（严格按 7:2:1）
        n_val = int(n_total * VAL_RATIO)
        n_test = int(n_total * TEST_RATIO)
        # 剩余的全部留在 train，保证总数 100%
        n_train = n_total - n_val - n_test

        # 划分图片
        val_imgs = img_files[:n_val]
        test_imgs = img_files[n_val: n_val + n_test]
        # train_imgs = img_files[n_val + n_test :] （留在原文件夹，无需操作）

        # 剪切移动到 val（从 train 剪切，原 train 不再保留）
        for f in val_imgs:
            shutil.move(str(cls_src_path / f), str(target_val_dir / cls / f))
            total_val += 1

        # 剪切移动到 test（从 train 剪切，原 train 不再保留）
        for f in test_imgs:
            shutil.move(str(cls_src_path / f), str(target_test_dir / cls / f))
            total_test += 1

        print(f"📊 类别 {cls:8s} | 总{n_total}张 → train:{n_train} / val:{n_val} / test:{n_test}")

    # 5. 最终统计
    total_all = total_val + total_test + (sum(len(os.listdir(src_train_dir / cls)) for cls in classes))
    print("\n" + "=" * 60)
    print("✅ 数据集划分完成！（train/val/test 完全无交集）")
    print(f"📈 总图片数：{total_all}")
    print(f"📂 训练集(train)：{total_all - total_val - total_test} 张（{TRAIN_RATIO * 100:.0f}%，留在原文件夹）")
    print(f"📂 验证集(val)：{total_val} 张（{VAL_RATIO * 100:.0f}%，从 train 剪切）")
    print(f"📂 测试集(test)：{total_test} 张（{TEST_RATIO * 100:.0f}%，从 train 剪切）")
    print("=" * 60)


if __name__ == "__main__":
    split_dataset_move()