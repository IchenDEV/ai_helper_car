#!/usr/bin/env python

from __future__ import print_function
import sys
import os
from argparse import ArgumentParser, SUPPRESS
import cv2
import time
import logging as log

from openvino.inference_engine import IENetwork, IECore


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-m", "--model", help="Required. Path to an .xml file with a trained model.",
                      required=True, type=str)
    args.add_argument("-i", "--input",
                      help="Required. Path to video file or image. 'cam' for capturing video stream from camera",
                      required=True, type=str)
    args.add_argument("-d", "--device",
                      help="Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL or MYRIAD is "
                           "acceptable. The demo will look for a suitable plugin for device specified. "
                           "Default value is CPU", default="MYRIAD", type=str)
    args.add_argument("-s","--sync", help="Optional. Synchronous or not", default=True,action='store_false')
    args.add_argument("--labels", help="Optional. Path to labels mapping file", default=None, type=str)
    args.add_argument("-pt", "--prob_threshold", help="Optional. Probability threshold for detections filtering",
                      default=0.5, type=float)

    return parser

                    # xmin = int(obj[3] * frame_w)
                    # ymin = int(obj[4] * frame_h)
                    # xmax = int(obj[5] * frame_w)
                    # ymax = int(obj[6] * frame_h)
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
    if box_1_area<box_2_area:
        box_1_area,box_2_area=box_2_area,box_1_area
    if area_of_union <= 0.0 :
        retval = 0.0
    else:
        retval = (area_of_overlap / area_of_union)
    if area_of_union/box_1_area >0.85 and area_of_overlap/box_2_area >0.85:
        retval = 0.5
    return retval

def main():
    fps = ""
    detectfps = ""
    framecount = 0
    detectframecount = 0
    time1 = 0
    time2 = 0
    LABELS = ('background','person')
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()
    model_xml = args.model

    model_bin = os.path.splitext(model_xml)[0] + ".bin"

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

    if args.input == 'cam':
        input_stream = -1
    else:
        input_stream = args.input
    cap = cv2.VideoCapture(input_stream)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 512)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 512)
    assert cap.isOpened(), "Can't open " + input_stream

    if args.labels:
        with open(args.labels, 'r') as f:
            labels_map = [x.strip() for x in f]
    else:
        labels_map = None

    cur_request_id = 0
    next_request_id = 1
    is_async_mode = args.sync

    if is_async_mode:
        ret, frame = cap.read()
        frame_h, frame_w = frame.shape[:2]
        log.info("Starting inference in async mode...")
    else :
        log.info("Starting inference in sync mode...")

    print("To close the application, press 'CTRL+C' here or switch to the output window and press q ")
    

    while cap.isOpened():
        t1 = time.perf_counter()
        if is_async_mode:
            ret, next_frame = cap.read()
        else:
            ret, frame = cap.read()
            if ret:
                frame_h, frame_w = frame.shape[:2]
        if not ret:
            break  # abandons the last frame in case of async_mode
        # 同步处理方式
        if is_async_mode:
            in_frame = cv2.resize(next_frame, (w, h))
            in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
            in_frame = in_frame.reshape((n, c, h, w))
            feed_dict[input_blob] = in_frame
            exec_net.start_async(request_id=next_request_id, inputs=feed_dict)
        else:
            in_frame = cv2.resize(frame, (w, h))
            in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
            in_frame = in_frame.reshape((n, c, h, w))
            feed_dict[input_blob] = in_frame
            exec_net.start_async(request_id=cur_request_id, inputs=feed_dict)
        if exec_net.requests[cur_request_id].wait(-1) == 0:
            # inf_end = time.time()
            detectframecount += 1
            # det_time = inf_end - inf_start
            # Parse detection results of the current request
            res = exec_net.requests[cur_request_id].outputs[out_blob]
            # print(res.shape)
            objects = []
            for obj in res[0][0]:
                # Draw only objects when probability more than specified threshold
                if obj[2] > args.prob_threshold:
                    # ymin = int(obj[4] * frame_h)
                    # ymax = int(obj[6] * frame_h)
                    # y_h = ymax - ymin
                    objects.append(obj)
            objlen = len(objects)
            # print ("objlen:",objlen)
            for i in range(objlen):
                for j in range(i + 1, objlen):
                    iou = IntersectionOverUnion(objects[i], objects[j]) 
                    # print("iou的值： ",iou)
                    if (iou >= 0.4):
                        if objects[i][2] < objects[j][2]:
                            objects[i], objects[j] = objects[j], objects[i]
                        objects[j][2] = 0.0
            for obj in objects:
                # Draw only objects when probability more than specified threshold
                if obj[2] > args.prob_threshold:      
                    xmin = int(obj[3] * frame_w)
                    ymin = int(obj[4] * frame_h)
                    xmax = int(obj[5] * frame_w)
                    ymax = int(obj[6] * frame_h)
                    class_id = int(obj[1])
                    label_text = LABELS[class_id] + ' (' + str(round(obj[2] * 100, 1)) + '%)'
                    # 画出矩阵框
                    box_color = (255, 128, 0)
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), box_color, 2)
                    # 画文字框框
                    label_background_color = (125, 175, 75)
                    label_text_color = (255, 255, 255)
                    label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
                    label_xmin = xmin
                    label_ymin = ymin - label_size[1]
                    if (label_ymin < 1):
                        label_ymin = 1
                    label_xmax = label_xmin + label_size[0]
                    label_ymax = label_ymin + label_size[1]
                    cv2.rectangle(frame, (label_xmin - 1, label_ymin - 1), (label_xmax + 1, label_ymax + 1), label_background_color,-1)
                    cv2.putText(frame, label_text, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_text_color, 1)
                


            async_mode_message = "Async." if is_async_mode else "Sync."

            cv2.putText(frame, fps,(int(frame_w-170),15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (38,0,255), 1, cv2.LINE_AA)
            cv2.putText(frame, detectfps, (int(frame_w-170),30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (38,0,255), 1, cv2.LINE_AA)

            cv2.putText(frame, async_mode_message, (10, int(frame_h - 20)), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                        (10, 10, 200), 1,cv2.LINE_AA)

        cv2.namedWindow('Detection Results',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Detection Results', 464,464)
        cv2.imshow("Detection Results", frame)

        framecount += 1
        if framecount >= 15:
            fps       = "(Playback) {:.1f} FPS".format(time1/15)
            detectfps = "(Detection) {:.1f} FPS".format(detectframecount/time2)
            framecount = 0
            detectframecount = 0
            time1 = 0
            time2 = 0
        t2 = time.perf_counter()
        elapsedTime = t2-t1
        time1 += 1/elapsedTime
        time2 += elapsedTime

        if is_async_mode:
            cur_request_id, next_request_id = next_request_id, cur_request_id
            frame = next_frame
            frame_h, frame_w = frame.shape[:2]

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    sys.exit(main() or 0)
