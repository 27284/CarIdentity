import cv2
import os
import sys
import numpy as np
import tensorflow as tf
import uuid

# 车牌字符对照表
char_table = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
              'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '川', '鄂', '赣', '甘', '贵',
              '桂', '黑', '沪', '冀', '津', '京', '吉', '辽', '鲁', '蒙', '闽', '宁', '青', '琼', '陕', '苏', '晋',
              '皖', '湘', '新', '豫', '渝', '粤', '云', '藏', '浙']

# 全局配置
car_plate_w, car_plate_h = 136, 36
char_w, char_h = 20, 20
# 获取当前脚本所在目录（code文件夹）
current_script_dir = os.path.dirname(os.path.abspath(__file__))
# 定位到项目根目录（CarPlateIdentity-master）
project_root = os.path.dirname(current_script_dir)
# 拼接输出目录：项目根目录/carIdentityData/opencv_output
OUTPUT_ROOT = os.path.join(project_root, "carIdentityData", "opencv_output")
os.makedirs(OUTPUT_ROOT, exist_ok=True)


def verify_scale(rotate_rect):
    aspect = 4
    min_area = 10 * (10 * aspect)
    max_area = 150 * (150 * aspect)
    min_aspect = aspect * 0.6
    max_aspect = aspect * 1.4
    if rotate_rect[1][0] == 0 or rotate_rect[1][1] == 0:
        return False
    r = rotate_rect[1][0] / rotate_rect[1][1]
    r = max(r, 1 / r)
    area = rotate_rect[1][0] * rotate_rect[1][1]
    if min_area < area < max_area and min_aspect < r < max_aspect:
        return True
    return False


def img_Transform(car_rect, image):
    img_h, img_w = image.shape[:2]
    rect_w, rect_h = car_rect[1][0], car_rect[1][1]
    angle = car_rect[2]
    if angle == 0 or (angle == -90 and rect_w < rect_h):
        if angle == -90:
            rect_w, rect_h = rect_h, rect_w
        car_img = image[int(car_rect[0][1] - rect_h / 2):int(car_rect[0][1] + rect_h / 2),
                  int(car_rect[0][0] - rect_w / 2):int(car_rect[0][0] + rect_w / 2)]
        return car_img
    car_rect = (car_rect[0], (rect_w, rect_h), angle)
    box = cv2.boxPoints(car_rect)
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
    if left_point[1] <= right_point[1]:
        new_right_point = [right_point[0], heigth_point[1]]
        pts1 = np.float32([left_point, heigth_point, right_point])
        pts2 = np.float32([left_point, heigth_point, new_right_point])
        M = cv2.getAffineTransform(pts1, pts2)
        dst = cv2.warpAffine(image, M, (round(img_w * 2), round(img_h * 2)))
        car_img = dst[int(left_point[1]):int(heigth_point[1]), int(left_point[0]):int(new_right_point[0])]
    else:
        new_left_point = [left_point[0], heigth_point[1]]
        pts1 = np.float32([left_point, heigth_point, right_point])
        pts2 = np.float32([new_left_point, heigth_point, right_point])
        M = cv2.getAffineTransform(pts1, pts2)
        dst = cv2.warpAffine(image, M, (round(img_w * 2), round(img_h * 2)))
        car_img = dst[int(right_point[1]):int(heigth_point[1]), int(new_left_point[0]):int(right_point[0])]
    return car_img


def pre_process(orig_img):
    gray_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
    blur_img = cv2.blur(gray_img, (3, 3))
    sobel_img = cv2.Sobel(blur_img, cv2.CV_16S, 1, 0, ksize=3)
    sobel_img = cv2.convertScaleAbs(sobel_img)
    hsv_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv_img[:, :, 0], hsv_img[:, :, 1], hsv_img[:, :, 2]
    blue_img = (((h > 26) & (h < 34)) | ((h > 100) & (h < 124))) & (s > 70) & (v > 70)
    blue_img = blue_img.astype('float32')
    mix_img = np.multiply(sobel_img, blue_img)
    mix_img = mix_img.astype(np.uint8)
    ret, binary_img = cv2.threshold(mix_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 5))
    close_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)
    return close_img


def verify_color(rotate_rect, src_image):
    img_h, img_w = src_image.shape[:2]
    mask = np.zeros([img_h + 2, img_w + 2], dtype=np.uint8)
    connectivity = 4
    loDiff, upDiff = 30, 30
    new_value = 255
    flags = connectivity | cv2.FLOODFILL_FIXED_RANGE | (new_value << 8) | cv2.FLOODFILL_MASK_ONLY
    rand_seed_num = 5000
    valid_seed_num = 200
    adjust_param = 0.1
    box_points = cv2.boxPoints(rotate_rect)
    box_x = sorted([p[0] for p in box_points])
    box_y = sorted([p[1] for p in box_points])
    adjust_x = int((box_x[2] - box_x[1]) * adjust_param)
    adjust_y = int((box_y[2] - box_y[1]) * adjust_param)
    col_range = [box_x[1] + adjust_x, box_x[2] - adjust_x]
    row_range = [box_y[1] + adjust_y, box_y[2] - adjust_y]
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
    hsv = cv2.cvtColor(src_image, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    cnt = 0
    for i in range(rand_seed_num):
        idx = np.random.choice(rand_seed_num, 1, replace=False)
        r, c = points_row[idx], points_col[idx]
        if (((26 < h[r, c] < 34) or (100 < h[r, c] < 124)) and s[r, c] > 70 and v[r, c] > 70):
            cv2.floodFill(src_image, mask, (c, r), (255, 255, 255), (loDiff,) * 3, (upDiff,) * 3, flags)
            cnt += 1
            if cnt >= valid_seed_num:
                break
    mask_pts = [(x - 1, y - 1) for y in range(1, img_h + 1) for x in range(1, img_w + 1) if mask[y, x] != 0]
    if not mask_pts:
        return False, None
    rect = cv2.minAreaRect(np.array(mask_pts))
    return verify_scale(rect), rect


def locate_carPlate(orig_img, pred_image):
    carPlate_list = []
    temp1 = orig_img.copy()
    temp2 = orig_img.copy()
    contours, _ = cv2.findContours(pred_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    for i, cnt in enumerate(contours):
        cv2.drawContours(temp1, [cnt], -1, (0, 255, 255), 2)
        rect = cv2.minAreaRect(cnt)
        if not verify_scale(rect):
            continue
        ok, rect2 = verify_color(rect, temp2)
        if not ok or rect2 is None:
            continue
        plate = img_Transform(rect2, temp2)
        plate = cv2.resize(plate, (car_plate_w, car_plate_h))
        carPlate_list.append(plate)
        box = cv2.boxPoints(rect2).astype(int)
        for k in range(4):
            cv2.line(temp1, tuple(box[k]), tuple(box[(k + 1) % 4]), (255, 0, 0), 2)
    cv2.imshow('contour', temp1)
    return carPlate_list


def horizontal_cut_chars(plate):
    char_addr_list = []
    area_left, area_right, char_left, char_right = 0, 0, 0, 0
    img_w = plate.shape[1]
    img_h = plate.shape[0]

    def getColSum(img, col):
        return sum(round(img[i, col] / 255) for i in range(img.shape[0]))

    h_count = [sum(1 for c in range(img_w) if plate[r, c] == 255) for r in range(img_h)]
    h_proj = []
    temp_len = 0
    start = 0
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
    if not h_proj:
        return []
    max_idx = np.argmax([e - s for s, e in h_proj])
    t, b = h_proj[max_idx]
    if (b - t) / img_h < 0.5:
        return []

    sum_col = sum(getColSum(plate, c) for c in range(img_w))
    col_limit = 0
    char_wid_min = round(img_w / 12)
    char_wid_max = round(img_w / 5)
    is_char = False

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
    if area_right < char_left:
        ar, cr = img_w, img_w
        aw, cw = ar - area_left, cr - char_left
        if char_wid_min < aw < char_wid_max:
            char_addr_list.append((area_left, ar, cw))

    char_imgs = []
    for i, (l, r, _) in enumerate(char_addr_list):
        char_img = plate[t:b + 1, l:r]
        char_img = cv2.resize(char_img, (char_w, char_h))
        char_imgs.append(char_img)
    return char_imgs


def get_chars(car_plate, output_dir):
    img_h, img_w = car_plate.shape[:2]
    h_count = [sum(1 for c in range(img_w) if car_plate[r, c] == 255) for r in range(img_h)]
    h_proj = []
    temp_len = 0
    start = 0
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
    if not h_proj:
        return []
    max_idx = np.argmax([e - s for s, e in h_proj])
    t, b = h_proj[max_idx]
    if (b - t) / img_h < 0.5:
        return []

    plate_crop = car_plate[t:b + 1, :]
    char_imgs = horizontal_cut_chars(plate_crop)
    # 逐字符保存
    for i, char_img in enumerate(char_imgs):
        char_path = os.path.join(output_dir, f"char_{i}.jpg")
        cv2.imwrite(char_path, char_img)
        print(f"✅ 已保存字符：{char_path}")
    return char_imgs


# ========== 保存二值化车牌 ==========
def extract_char(car_plate, output_dir):
    gray = cv2.cvtColor(car_plate, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # 转3通道彩色，正常显示
    binary_color = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    bin_save_path = os.path.join(output_dir, "binary_plate.jpg")
    cv2.imwrite(bin_save_path, binary_color)
    print(f"✅ 3. 已保存 binary_plate.jpg")

    return get_chars(binary, output_dir)


def cnn_select_carPlate(plate_list, model_path):
    if not plate_list:
        return False, None
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
        return False, None
    return True, plate_list[best_idx]


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


if __name__ == '__main__':
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    plate_model_path = os.path.abspath(
        os.path.join(cur_dir, '../carIdentityData/model/plate_recongnize/model.ckpt-510.meta'))
    char_model_path = os.path.abspath(
        os.path.join(cur_dir, '../carIdentityData/model/char_recongnize/model.ckpt-810.meta'))
    img_path = os.path.abspath(os.path.join(cur_dir, '../carIdentityData/pictures/1.jpg'))

    print("车牌模型路径：", plate_model_path)
    print("字符模型路径：", char_model_path)
    print("图片路径：", img_path)
    print("输出根目录：", OUTPUT_ROOT)

    img = cv2.imread(img_path)
    if img is None:
        print("❌ 图片不存在或无法读取")
        sys.exit()

    pred_img = pre_process(img)
    plates = locate_carPlate(img, pred_img)
    if not plates:
        print("未检测到车牌")
        sys.exit()

    ok, plate = cnn_select_carPlate(plates, plate_model_path)
    if not ok:
        print("未检测到车牌")
        sys.exit()
    cv2.imshow('cnn_plate', plate)

    # 新建唯一文件夹
    folder_name = str(uuid.uuid4())[:8]
    output_dir = os.path.join(OUTPUT_ROOT, folder_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"✅ 已创建输出文件夹：{output_dir}")

    # 1. 先保存 car
    cv2.imwrite(os.path.join(output_dir, "car.jpg"), img)
    print("✅ 1. 已保存 car.jpg")

    # 2. 再保存 plate
    cv2.imwrite(os.path.join(output_dir, "plate.jpg"), plate)
    print("✅ 2. 已保存 plate.jpg")

    # 3. 自动保存 binary_plate + 4. 保存字符
    chars = extract_char(plate, output_dir)

    if not chars:
        print("未分割到字符")
        sys.exit()

    # CNN识别结果
    result = cnn_recongnize_char(chars, char_model_path)
    print("\n✅ 最终识别车牌：", ''.join(result))

    cv2.waitKey(0)
    cv2.destroyAllWindows()