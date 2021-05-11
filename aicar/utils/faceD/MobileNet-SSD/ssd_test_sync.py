#!/usr/bin/env python

from __future__ import print_function
import sys
import os
from argparse import ArgumentParser, SUPPRESS
import cv2
import time
import logging as log

from openvino.inference_engine import IENetwork, IECore
import shutil
import numpy as np

def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-m", "--model", help="Required. Path to an .xml file with a trained model.",
                      required=True, type=str)
    args.add_argument("-d", "--device",
                      help="Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL or MYRIAD is "
                           "acceptable. The demo will look for a suitable plugin for device specified. "
                           "Default value is CPU", default="MYRIAD", type=str)
    args.add_argument("-pt", "--prob_threshold", help="Optional. Probability threshold for detections filtering",
                      default=0.4, type=float)
    return parser

def IntersectionOverUnion(box_1, box_2):
    width_of_overlap_area = min(box_1[5], box_2[5]) - max(box_1[3], box_2[3])
    height_of_overlap_area = min(box_1[6], box_2[6]) - max(box_1[4], box_2[4])
    area_of_overlap = 0.0
    if (width_of_overlap_area < 0.0 or height_of_overlap_area < 0.0):
        area_of_overlap = 0.0
    else:
        area_of_overlap = width_of_overlap_area * height_of_overlap_area
    box_1_area = (box_1[6] - box_1[4])  * (box_1[5] - box_1[3])
    box_2_area = (box_2[6] - box_2[4])  * (box_2[5] - box_2[3])
    area_of_union = box_1_area + box_2_area - area_of_overlap
    retval = 0.0
    if area_of_union <= 0.0 :
        retval = 0.0
    else:
        retval = (area_of_overlap / area_of_union)
    return retval

def main():
    args = build_argparser().parse_args()

    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    model_xml = args.model
    model_bin = os.path.splitext(model_xml)[0] + ".bin"
    file_name=os.path.splitext(model_xml)[0].split('/')[-2]

    predicted_dir_path = 'mAP/'+file_name+'_sync/predicted'
    if os.path.exists(predicted_dir_path): shutil.rmtree(predicted_dir_path) #递归删除所有文件
    os.makedirs(predicted_dir_path)

    log.info("Creating Inference Engine...")
    ie = IECore()
    log.info("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
    net = IENetwork(model=model_xml, weights=model_bin)

    img_info_input_blob = None
    feed_dict = {}
    for blob_name in net.inputs:
        if len(net.inputs[blob_name].shape) == 4:
            input_blob = blob_name
        elif len(net.inputs[blob_name].shape) == 2:
            img_info_input_blob = blob_name
        else:
            raise RuntimeError("Unsupported {}D input layer '{}'. Only 2D and 4D input layers are supported"
                                .format(len(net.inputs[blob_name].shape), blob_name))

    assert len(net.outputs) == 1, "Demo supports only single output topologies"

    out_blob = next(iter(net.outputs))
    log.info("Loading IR to the plugin...")
    exec_net = ie.load_network(network=net, num_requests=2, device_name=args.device)
    # 读取图片并进行预处理
    n, c, h, w = net.inputs[input_blob].shape
    if img_info_input_blob:
        feed_dict[img_info_input_blob] = [h, w, 1]

    cur_request_id = 0

    print("To close the application, press 'CTRL+C' here or switch to the output window and press q ")

    
    start=time.perf_counter()
    # print("start:",start)
    with open("/home/pi/Desktop/MobileNet-SSD/VOCdevkit/2007_test.txt", 'r') as annotation_file:
        for num, line in enumerate(annotation_file):
            annotation = line.strip().split()
            image_path = annotation[0]
            image_name = image_path.split('/')[-1]
            image = cv2.imread(image_path)
            frame_h, frame_w = image.shape[0:2]
            if(num % 100 == 0):
                print('num:%s'%num)
            

            # print('=> predict result of %s:' % image_name)
            predict_result_path = os.path.join(predicted_dir_path, str(num) + '.txt')
            # 预测过程
            in_frame= cv2.resize(image,(320,320))
            in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
            in_frame = in_frame.reshape((n, c, h, w))
            feed_dict[input_blob] = in_frame
            exec_net.start_async(request_id=cur_request_id, inputs=feed_dict)
            if exec_net.requests[cur_request_id].wait(-1) == 0:
                res = exec_net.requests[cur_request_id].outputs[out_blob]
                objects = []
                for obj in res[0][0]:
                    # Draw only objects when probability more than specified threshold
                    if obj[2] > args.prob_threshold:
                        objects.append(obj)
                objlen = len(objects)
                for i in range(objlen):
                    for j in range(i + 1, objlen):
                        iou = IntersectionOverUnion(objects[i], objects[j]) 
                        if (iou >= 0.4):
                            if objects[i][2] < objects[j][2]:
                                objects[i], objects[j] = objects[j], objects[i]
                            objects[j][2] = 0.0
                with open(predict_result_path, 'w') as f:
                    for obj in objects:
                        if obj[2] > args.prob_threshold:      
                            xmin = int(obj[3] * frame_w)
                            ymin = int(obj[4] * frame_h)
                            xmax = int(obj[5] * frame_w)
                            ymax = int(obj[6] * frame_h)
                            class_id = int(obj[1])
                            score = '%.4f' % obj[2]
                            list1=[class_id, score, xmin, ymin, xmax, ymax]
                            bbox_mess = ' '.join('%s' %id for id in list1) + '\n'
                            f.write(bbox_mess)
                            # print('\t' + str(bbox_mess).strip())
    end=time.perf_counter()
    # print("\nend:",end)
    duration=end-start
    print("\nThe duration is %.1f s" % duration)

if __name__ == '__main__':
    sys.exit(main() or 0)
