# -*- coding: utf-8 -*-
from src import detect_faces, show_bboxes
from PIL import Image
import cv2
import numpy as np
from src.align_trans import get_reference_facial_points, warp_and_crop_face
import time
from facenet import FaceNet
import math
import os

parent = os.path.dirname(os.path.realpath(__file__))


# 人脸特征距离度量,余弦距离
def cosine_distance(a, b):
    if a.shape != b.shape:
        raise RuntimeError("array {} shape not match {}".format(a.shape, b.shape))
    if a.ndim == 1:
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
    elif a.ndim == 2:
        a_norm = np.linalg.norm(a, axis=1, keepdims=True)
        b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    else:
        raise RuntimeError("array dimensions {} not right".format(a.ndim))
    similiarity = np.dot(a, b.T) / (a_norm * b_norm)

    return similiarity


# 核心函数
def get_features(facenet, img, origin_img, scale=1.):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # b, g, r = cv2.split(img)
    # img2 = cv2.merge([r, g, b])
    start = time.time()
    # 1.人脸检测,关键点定位
    bounding_boxes, landmarks = detect_faces(img_rgb, min_face_size=25.0)
    end = time.time()
    print("检测mtcnn运行时间:%.4f秒" % (end - start))
    faces = []
    img_cv2 = origin_img  # np.array(img)[...,::-1]
    for i in range(len(bounding_boxes)):
        box = bounding_boxes[i][:4].astype(np.int32).tolist()
        for idx, coord in enumerate(box[:2]):
            if coord > 1:
                box[idx] -= 1
        if box[2] + 1 < img_cv2.shape[1]:
            box[2] += 1
        if box[3] + 1 < img_cv2.shape[0]:
            box[3] += 1

        box[0] = int(box[0] / scale)
        box[1] = int(box[1] / scale)
        box[2] = int(box[2] / scale)
        box[3] = int(box[3] / scale)

        face = img_cv2[box[1]:box[3], box[0]:box[2]]
        face_height, face_width, face_channel = face.shape
        landmark = landmarks[i]
        facial5points = []
        # 防止人脸关键点预测出错
        for j in range(5):
            x = int(landmark[j] / scale) - box[0]
            if x < 0:
                x = 0
            elif x > face_width:
                x = face_width
            y = int(landmark[j + 5] / scale) - box[1]
            if y < 0:
                y = 0
            elif y > face_width:
                y = face_height
            point = [x, y]
            facial5points.append(point)
        # 2.仿射变换，根据人脸关键点进行人脸对其
        dst_img = warp_and_crop_face(face, facial5points)

        # 3.根据仿射变换的到人脸进行人脸识别
        dst_img_rgb = cv2.cvtColor(dst_img, cv2.COLOR_BGR2RGB)
        face_input = (dst_img_rgb / 255. - 0.5) / 0.5
        start = time.time()
        feature = facenet.get_feature(face_input)
        end = time.time()
        print("计算棒FaceNet运行时间:%.4f秒" % (end - start))
        faces.append([box, feature])
    return faces


# temper=45
def getDetectImage(img):  # 45-90

    min_size = get_pitsize()
    print('data in pit_size.txt:' + str(min_size))

    height, width, c = img.shape
    min_length = min(height, width)
    scale = min_size / min_length
    sw, sh = math.ceil(width * scale), math.ceil(height * scale)
    # img = image.resize((sw, sh), Image.BILINEAR)
    input = cv2.resize(img, (sw, sh))
    return input, scale


def get_pitsize():
    f = open(parent + '/pit_size.txt', 'r')
    minsize = f.read()
    f.close()
    return int(minsize)


def draw_text(frame, text, coordinate, line_color=(0, 0, 255), normalized=False):
    x1 = coordinate[0]
    y1 = coordinate[1]
    x2 = coordinate[2]
    y2 = coordinate[3]

    if normalized:
        h = frame.shape[0]
        w = frame.shape[1]

        x1 = int(x1 * w)
        x2 = int(x2 * w)

        y1 = int(y1 * h)
        y2 = int(y2 * h)

    font = cv2.FONT_HERSHEY_SIMPLEX
    bottom_left_corner_of_text = (x2, y1 + 10)
    font_scale = 0.8
    font_color = line_color
    line_type = 2

    cv2.putText(frame,
                text,
                bottom_left_corner_of_text,
                font,
                font_scale,
                font_color,
                line_type)


def draw_rectangle(frame, coordinate, line_color=(0, 255, 124), normalized=False):
    x1 = coordinate[0]
    y1 = coordinate[1]
    x2 = coordinate[2]
    y2 = coordinate[3]

    if normalized:
        h = frame.shape[0]
        w = frame.shape[1]

        x1 = int(x1 * w)
        x2 = int(x2 * w)

        y1 = int(y1 * h)
        y2 = int(y2 * h)

    cv2.rectangle(frame, (x1, y1), (x2, y2), line_color, 2)


# if __name__ == '__main__':
def start_face_detector(Vshow):
    # 判断是否是同一人的阈值
    same_person_threshold = 0.4
    facenet = FaceNet(parent + "/facenet/res20_prelu.xml", device="CPU")  # CPU
    candidate_features = []
    candidate_counter = 0
    cv2.namedWindow("FaceDemo")
    print("1. 正在获取人脸识别文件夹中的人脸特征")

    dirs = os.listdir(parent + "/candidate_person")
    for file_name in dirs:
        if '.py' in file_name or '.md' in file_name or '.jpg' in file_name:
            continue
        else:
            candidate_counter += 1
            for file in os.listdir(parent + "\\candidate_person" + '\\' + file_name):
                if "origin." in file and ".jpg" in file:
                    mid_str = file.replace("origin.", "")
                    mid_str = mid_str.replace(".jpg", "")
                    print("已获取No.%s.jpg文件" % mid_str)
                    image_path = os.path.join(parent + "\\candidate_person\\" + file_name, file)
                    print(image_path)
                    candidate_img = cv2.imread(image_path)
                    print(candidate_img)
                    candidate_img_input, scale = getDetectImage(candidate_img)
                    candidate_feature = get_features(facenet, candidate_img_input, candidate_img, scale)
                    if len(candidate_feature) > 0:
                        candidate = candidate_feature[0]
                        bbox = candidate[0]
                        face = candidate_img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                        cv2.imwrite(parent + "/candidate_person/" + file_name + "/candidate.%s.jpg" % mid_str, face)
                        candidate_features.append([mid_str, candidate])

    print("数据库中共有%d人" % candidate_counter)
    if len(candidate_features) == 0:
        exit(-1)
    print("2. 开始人脸检测并识别")
    # cv2.VideoCapture(0)代表调取摄像头资源，其中0代表电脑摄像头，1代表外接摄像头(usb摄像头)

    count = 0
    count = count + 1

    print(Vshow.shape)
    test_img_input, scale = getDetectImage(Vshow)

    start = time.time()
    test_features = get_features(facenet, test_img_input, Vshow, scale)
    end = time.time()
    total_faces = []
    for i in range(len(test_features)):
        bbox = test_features[i][0]
        bbox[0] = int(bbox[0])
        bbox[1] = int(bbox[1])
        bbox[2] = int(bbox[2])
        bbox[3] = int(bbox[3])
        face = Vshow[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        # 人脸特征提取
        feature = test_features[i][1]
        # 相似的计算
        top_sim = -1.0
        top_name = ""
        for candidate_feature in candidate_features:
            mid_str = candidate_feature[0]
            candidate = candidate_feature[1]
            sim = cosine_distance(candidate[1][0], feature[0])
            if sim > top_sim:
                top_sim = sim
                if top_sim > same_person_threshold:
                    top_name = mid_str
        total_faces.append([bbox, top_name, top_sim])
        print("%d_sim" % (i), top_sim)
    for i in range(len(total_faces)):
        faces = total_faces[i]
        bbox = faces[0]
        show_name = faces[1]
        top_sim = faces[2]
        if show_name != "":
            draw_rectangle(Vshow, bbox, (255, 0, 0))
            showname = show_name[:show_name.find("_")]
            draw_text(Vshow, "%s:%s" % (showname, str(top_sim)), bbox)
            return True
        else:
            draw_rectangle(Vshow, bbox)
            return False


if __name__ == '__main__':
    start_face_detector()
