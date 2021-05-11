#! /usr/bin/env python

import cv2
import os
import shutil
import numpy as np

ground_truth_dir_path = 'mAP/ground-truth'
if os.path.exists(ground_truth_dir_path): shutil.rmtree(ground_truth_dir_path) 
os.mkdir(ground_truth_dir_path)

with open('/home/pi/Desktop/MobileNet-SSD/VOCdevkit/2007_test.txt', 'r') as annotation_file:
    for num, line in enumerate(annotation_file):
        annotation = line.strip().split()
        image_path = annotation[0]
        image_name = image_path.split('\\')[-1]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bbox_data_gt = np.array([list(map(int, box.split(','))) for box in annotation[1:]])

        if len(bbox_data_gt) == 0:
            bboxes_gt=[]
            classes_gt=[]
        else:
            bboxes_gt, classes_gt = bbox_data_gt[:, :4], bbox_data_gt[:, 4]
        ground_truth_path = os.path.join(ground_truth_dir_path, str(num) + '.txt')

        print('=> ground truth of %s:' % image_name)
        num_bbox_gt = len(bboxes_gt)
        with open(ground_truth_path, 'w') as f:
            for i in range(num_bbox_gt):
                class_name = '1'
                xmin, ymin, xmax, ymax = list(map(str, bboxes_gt[i]))
                #将序列中的元素以指定的字符连接生成一个新的字符串，' '表示用空格做分隔符
                bbox_mess = ' '.join([class_name, xmin, ymin, xmax, ymax]) + '\n'
                # 写真值框
                f.write(bbox_mess)
                print('\t' + str(bbox_mess).strip())
