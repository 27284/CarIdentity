import cv2
import os
import sys
import numpy as np
import tensorflow as tf

# 车牌字符对照表：0-9、A-Z、各省简称，共65个字符
char_table = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
              'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '川', '鄂', '赣', '甘', '贵',
              '桂', '黑', '沪', '冀', '津', '京', '吉', '辽', '鲁', '蒙', '闽', '宁', '青', '琼', '陕', '苏', '晋',
              '皖', '湘', '新', '豫', '渝', '粤', '云', '藏', '浙']


# 直方图均衡化：增强图像对比度
def hist_image(img):
    assert img.ndim == 2  # 确保输入是灰度图
    hist = [0 for i in range(256)]
    img_h, img_w = img.shape[0], img.shape[1]

    # 统计每个灰度值出现的次数
    for row in range(img_h):
        for col in range(img_w):
            hist[img[row, col]] += 1

    # 计算概率与累积分布
    p = [hist[n] / (img_w * img_h) for n in range(256)]
    p1 = np.cumsum(p)

    # 重新映射像素
    for row in range(img_h):
        for col in range(img_w):
            v = img[row, col]
            img[row, col] = p1[v] * 255
    return img

# 查找车牌大致区域（上下左右边界）
def find_board_area(img):
    assert img.ndim == 2
    img_h, img_w = img.shape[0], img.shape[1]
    top, bottom, left, right = 0, img_h, 0, img_w
    flag = False
    h_proj = [0 for i in range(img_h)]
    v_proj = [0 for i in range(img_w)]

    # 水平投影找上下边界
    for row in range(round(img_h * 0.5), round(img_h * 0.8), 3):
        for col in range(img_w):
            if img[row, col] == 255:
                h_proj[row] += 1
        if not flag and h_proj[row] > 12:
            flag = True
            top = row
        if flag and row > top + 8 and h_proj[row] < 12:
            bottom = row
            flag = False

    # 垂直投影找左边界
    for col in range(round(img_w * 0.3), img_w, 1):
        for row in range(top, bottom, 1):
            if img[row, col] == 255:
                v_proj[col] += 1
        if not flag and (v_proj[col] > 10 or (col > 0 and v_proj[col] - v_proj[col - 1] > 5)):
            left = col
            break
    return left, top, 120, bottom - top - 10

# 判断矩形是否符合车牌比例（宽高比、面积）
def verify_scale(rotate_rect):
    error = 0.4
    aspect = 4  # 车牌典型宽高比
    min_area = 10 * (10 * aspect)
    max_area = 150 * (150 * aspect)
    min_aspect = aspect * (1 - error)
    max_aspect = aspect * (1 + error)
    theta = 30

    # 宽高为0直接排除
    if rotate_rect[1][0] == 0 or rotate_rect[1][1] == 0:
        return False

    r = rotate_rect[1][0] / rotate_rect[1][1]
    r = max(r, 1 / r)
    area = rotate_rect[1][0] * rotate_rect[1][1]

    # 面积与比例判断
    if min_area < area < max_area and min_aspect < r < max_aspect:
        # 倾斜角度限制
        if ((rotate_rect[1][0] < rotate_rect[1][1] and -90 <= rotate_rect[2] < -(90 - theta)) or
                (rotate_rect[1][1] < rotate_rect[1][0] and -theta < rotate_rect[2] <= 0)):
            return True
    return False

# 车牌倾斜校正 + 透视变换
def img_Transform(car_rect, image):
    img_h, img_w = image.shape[:2]
    rect_w, rect_h = car_rect[1][0], car_rect[1][1]
    angle = car_rect[2]

    # 无倾斜直接裁剪
    if angle == 0 or (angle == -90 and rect_w < rect_h):
        if angle == -90:
            rect_w, rect_h = rect_h, rect_w
        car_img = image[int(car_rect[0][1] - rect_h / 2):int(car_rect[0][1] + rect_h / 2),
                  int(car_rect[0][0] - rect_w / 2):int(car_rect[0][0] + rect_w / 2)]
        return car_img

    # 有倾斜：获取最小外接矩形四个点
    car_rect = (car_rect[0], (rect_w, rect_h), angle)
    box = cv2.boxPoints(car_rect)

    # 定位四个角点
    heigth_point = right_point = [0, 0]
    left_point = low_point = [car_rect[0][0], car_rect[0][1]]
    for point in box:
        if left_point[0] > point[0]:
            left_point = point
        if low_point[1] > point[1]:
            low_point = point
        if heigth_point[1] < point[1]:
            heigth_point = point
        if right_point[0] < point[0]:
            right_point = point

    # 正角度校正
    if left_point[1] <= right_point[1]:
        new_right_point = [right_point[0], heigth_point[1]]
        pts1 = np.float32([left_point, heigth_point, right_point])
        pts2 = np.float32([left_point, heigth_point, new_right_point])
        M = cv2.getAffineTransform(pts1, pts2)
        dst = cv2.warpAffine(image, M, (round(img_w * 2), round(img_h * 2)))
        car_img = dst[int(left_point[1]):int(heigth_point[1]), int(left_point[0]):int(new_right_point[0])]

    # 负角度校正
    else:
        new_left_point = [left_point[0], heigth_point[1]]
        pts1 = np.float32([left_point, heigth_point, right_point])
        pts2 = np.float32([new_left_point, heigth_point, right_point])
        M = cv2.getAffineTransform(pts1, pts2)
        dst = cv2.warpAffine(image, M, (round(img_w * 2), round(img_h * 2)))
        car_img = dst[int(right_point[1]):int(heigth_point[1]), int(new_left_point[0]):int(right_point[0])]

    return car_img


# 图像预处理：灰度、模糊、边缘、颜色筛选、二值化、闭运算
def pre_process(orig_img):
    # 1. 转灰度图
    gray_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)

    # 2. 均值模糊去噪
    blur_img = cv2.blur(gray_img, (3, 3))

    # 3. Sobel水平边缘检测
    sobel_img = cv2.Sobel(blur_img, cv2.CV_16S, 1, 0, ksize=3)
    sobel_img = cv2.convertScaleAbs(sobel_img)

    # 4. HSV颜色空间：提取蓝色/黄色车牌区域
    hsv_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv_img[:, :, 0], hsv_img[:, :, 1], hsv_img[:, :, 2]

    # 黄色：26-34，蓝色：100-124，饱和度>70，亮度>70
    blue_img = (((h > 26) & (h < 34)) | ((h > 100) & (h < 124))) & (s > 70) & (v > 70)
    blue_img = blue_img.astype('float32')

    # 5. 边缘与颜色区域融合
    mix_img = np.multiply(sobel_img, blue_img)
    mix_img = mix_img.astype(np.uint8)

    # 6. 二值化（OTSU自动阈值）
    ret, binary_img = cv2.threshold(mix_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # 7. 闭运算：连接断裂区域
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 5))
    close_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)

    return close_img

# 漫水填充验证车牌区域（颜色+连通域）
def verify_color(rotate_rect, src_image):
    img_h, img_w = src_image.shape[:2]
    mask = np.zeros([img_h + 2, img_w + 2], dtype=np.uint8)
    connectivity = 4
    loDiff, upDiff = 30, 30
    new_value = 255
    flags = connectivity | cv2.FLOODFILL_FIXED_RANGE | (new_value << 8) | cv2.FLOODFILL_MASK_ONLY

    # 随机种子点
    rand_seed_num = 5000
    valid_seed_num = 200
    adjust_param = 0.1
    box_points = cv2.boxPoints(rotate_rect)

    # 计算种子点范围
    box_x = sorted([p[0] for p in box_points])
    box_y = sorted([p[1] for p in box_points])
    adjust_x = int((box_x[2] - box_x[1]) * adjust_param)
    adjust_y = int((box_y[2] - box_y[1]) * adjust_param)
    col_range = [box_x[1] + adjust_x, box_x[2] - adjust_x]
    row_range = [box_y[1] + adjust_y, box_y[2] - adjust_y]

    # 生成种子
    if ((col_range[1] - col_range[0]) / (box_x[3] - box_x[0]) < 0.4 or
            (row_range[1] - row_range[0]) / (box_y[3] - box_y[0]) < 0.4):
        points_row, points_col = [], []
        for i in range(2):
            pt1, pt2 = box_points[i], box_points[i + 2]
            x_adj = int(adjust_param * abs(pt1[0] - pt2[0]))
            y_adj = int(adjust_param * abs(pt1[1] - pt2[1]))
            pt1[0] += x_adj if pt1[0] <= pt2[0] else -x_adj
            pt2[0] -= x_adj if pt1[0] <= pt2[0] else -x_adj
            pt1[1] += y_adj if pt1[1] <= pt2[1] else -y_adj
            pt2[1] -= y_adj if pt1[1] <= pt2[1] else -y_adj
            xs = np.linspace(pt1[0], pt2[0], rand_seed_num // 2).astype(int)
            ys = np.linspace(pt1[1], pt2[1], rand_seed_num // 2).astype(int)
            points_col.extend(xs)
            points_row.extend(ys)
    else:
        points_row = np.random.randint(row_range[0], row_range[1], size=rand_seed_num)
        points_col = np.linspace(col_range[0], col_range[1], rand_seed_num).astype(int)

    # HSV判断背景色
    hsv = cv2.cvtColor(src_image, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    flood_img = src_image.copy()
    cnt = 0

    # 漫水填充
    for i in range(rand_seed_num):
        idx = np.random.choice(rand_seed_num, 1, replace=False)
        r, c = points_row[idx], points_col[idx]
        if (((26 < h[r, c] < 34) or (100 < h[r, c] < 124)) and s[r, c] > 70 and v[r, c] > 70):
            cv2.floodFill(src_image, mask, (c, r), (255, 255, 255), (loDiff,) * 3, (upDiff,) * 3, flags)
            cv2.circle(flood_img, (c, r), 2, (0, 0, 255), 2)
            cnt += 1
            if cnt >= valid_seed_num:
                break

    # 获取填充区域并判断是否符合车牌
    mask_pts = [(x - 1, y - 1) for y in range(1, img_h + 1) for x in range(1, img_w + 1) if mask[y, x] != 0]
    if not mask_pts:
        return False, None
    rect = cv2.minAreaRect(np.array(mask_pts))
    return verify_scale(rect), rect

# 车牌定位主函数：轮廓+筛选+校正
def locate_carPlate(orig_img, pred_image):
    carPlate_list = []
    temp1 = orig_img.copy()
    temp2 = orig_img.copy()

    # 查找轮廓
    contours, _ = cv2.findContours(pred_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]

    for i, cnt in enumerate(contours):
        cv2.drawContours(temp1, [cnt], -1, (0, 255, 255), 2)
        rect = cv2.minAreaRect(cnt)
        if not verify_scale(rect):
            continue

        # 漫水填充二次验证
        ok, rect2 = verify_color(rect, temp2)
        if not ok or rect2 is None:
            continue

        # 校正+缩放
        plate = img_Transform(rect2, temp2)
        plate = cv2.resize(plate, (car_plate_w, car_plate_h))
        carPlate_list.append(plate)

        # 绘制最终车牌框
        box = cv2.boxPoints(rect2).astype(int)
        for k in range(4):
            cv2.line(temp1, tuple(box[k]), tuple(box[(k + 1) % 4]), (255, 0, 0), 2)

    cv2.imshow('contour', temp1)
    return carPlate_list


# 字符垂直分割：左右切割
def horizontal_cut_chars(plate):
    char_addr_list = []
    area_left, area_right, char_left, char_right = 0, 0, 0, 0
    img_w = plate.shape[1]

    # 计算每列白色像素数量
    def getColSum(img, col):
        return sum(round(img[i, col] / 255) for i in range(img.shape[0]))

    sum_col = sum(getColSum(plate, c) for c in range(img_w))
    col_limit = 0
    char_wid_min = round(img_w / 12)
    char_wid_max = round(img_w / 5)
    is_char = False

    # 遍历每列
    for i in range(img_w):
        val = getColSum(plate, i)
        if val > col_limit:
            if not is_char:
                ar = round((i + char_right) / 2)
                aw, cw = ar - area_left, char_right - char_left
                if char_wid_min < aw < char_wid_max:
                    char_addr_list.append((area_left, ar, cw))
                char_left = i
                area_left = round((char_left + char_right) / 2)
                is_char = True
        else:
            if is_char:
                char_right = i - 1
                is_char = False

    # 处理最后一个字符
    if area_right < char_left:
        ar, cr = img_w, img_w
        aw, cw = ar - area_left, cr - char_left
        if char_wid_min < aw < char_wid_max:
            char_addr_list.append((area_left, ar, cw))
    return char_addr_list


# 字符水平切割+提取单个字符
def get_chars(car_plate):
    img_h, img_w = car_plate.shape[:2]
    h_proj = []
    temp_len = 0
    start = 0
    char_imgs = []

    # 水平投影
    h_count = [sum(1 for c in range(img_w) if car_plate[r, c] == 255) for r in range(img_h)]

    # 统计连续有效高度
    for r in range(img_h):
        ratio = h_count[r] / img_w
        if not (0.2 < ratio < 0.8):
            if temp_len > 0:
                h_proj.append((start, r - 1))
                temp_len = 0
            continue
        if h_count[r] > 0:
            if temp_len == 0:
                start = r
            temp_len += 1
        else:
            if temp_len > 0:
                h_proj.append((start, r - 1))
                temp_len = 0
    if temp_len > 0:
        h_proj.append((start, img_h - 1))

    # 取最长有效高度
    if not h_proj:
        return []
    max_idx = np.argmax([e - s for s, e in h_proj])
    t, b = h_proj[max_idx]
    if (b - t) / img_h < 0.5:
        return []

    # 垂直分割字符
    plate_crop = car_plate[t:b + 1, :]
    addrs = horizontal_cut_chars(plate_crop)
    for l, r, _ in addrs:
        char = car_plate[t:b + 1, l:r]
        char = cv2.resize(char, (char_w, char_h))
        char_imgs.append(char)
    return char_imgs


# 字符提取入口：灰度+二值化
def extract_char(car_plate):
    gray = cv2.cvtColor(car_plate, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return get_chars(binary)

# CNN：筛选真正的车牌
def cnn_select_carPlate(plate_list, model_path):
    if not plate_list:
        return False, plate_list[0] if plate_list else None
    g = tf.Graph()
    with tf.Session(graph=g) as sess:
        saver = tf.train.import_meta_graph(model_path)
        saver.restore(sess, tf.train.latest_checkpoint(os.path.dirname(model_path)))
        x = g.get_tensor_by_name('x_place:0')
        keep = g.get_tensor_by_name('keep_place:0')
        out = g.get_tensor_by_name('out_put:0')

        prob = tf.nn.softmax(out)
        pred = tf.argmax(prob, 1)
        maxp = tf.reduce_max(prob, 1)

        preds, probs = sess.run([pred, maxp], {x: np.array(plate_list), keep: 1.0})

    best_idx, best_p = -1, 0
    for i, p in enumerate(preds):
        if p == 1 and probs[i] > best_p:
            best_idx, best_p = i, probs[i]
    if best_idx == -1:
        return False, plate_list[0]
    return True, plate_list[best_idx]

# CNN：字符识别
def cnn_recongnize_char(img_list, model_path):
    if not img_list:
        return []
    g = tf.Graph()
    with tf.Session(graph=g) as sess:
        saver = tf.train.import_meta_graph(model_path)
        saver.restore(sess, tf.train.latest_checkpoint(os.path.dirname(model_path)))
        x = g.get_tensor_by_name('x_place:0')
        keep = g.get_tensor_by_name('keep_place:0')
        out = g.get_tensor_by_name('out_put:0')

        res = tf.argmax(tf.nn.softmax(out), 1)
        chars = sess.run(res, {x: np.array(img_list), keep: 1.0})
    return [char_table[c] for c in chars]


# 主函数：程序入口
if __name__ == '__main__':
    # 获取当前脚本所在绝对路径（解决所有路径错误）
    cur_dir = os.path.dirname(os.path.abspath(__file__))

    # 车牌/字符尺寸
    car_plate_w, car_plate_h = 136, 36
    char_w, char_h = 20, 20


    plate_model_path = os.path.abspath(os.path.join(cur_dir, '../carIdentityData/model/plate_recongnize/model.ckpt-510.meta'))
    char_model_path = os.path.abspath(os.path.join(cur_dir, '../carIdentityData/model/char_recongnize/model.ckpt-810.meta'))
    img_path = os.path.abspath(os.path.join(cur_dir, '../carIdentityData/pictures/1.jpg'))


    # 打印路径
    print("车牌模型路径：", plate_model_path)
    print("字符模型路径：", char_model_path)
    print("图片路径：", img_path)

    # 读取图片并校验
    img = cv2.imread(img_path)
    if img is None:
        print("❌ 图片不存在或无法读取")
        sys.exit()

    # 预处理
    pred_img = pre_process(img)

    # 定位车牌
    plates = locate_carPlate(img, pred_img)
    if not plates:
        print("未检测到车牌")
        sys.exit()

    # CNN筛选车牌
    ok, plate = cnn_select_carPlate(plates, plate_model_path)
    if not ok:
        print("未检测到车牌")
        sys.exit()
    cv2.imshow('cnn_plate', plate)

    # 字符分割
    chars = extract_char(plate)

    # 字符识别
    result = cnn_recongnize_char(chars, char_model_path)
    print("\n✅ 识别结果：", ''.join(result))

    cv2.waitKey(0)
    cv2.destroyAllWindows()