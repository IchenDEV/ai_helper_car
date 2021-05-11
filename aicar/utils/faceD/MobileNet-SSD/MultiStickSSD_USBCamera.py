import sys
if sys.version_info.major < 3 or sys.version_info.minor < 4:
    print("Please using python3.4 or greater!")
    sys.exit(1)
import numpy as np
import cv2, io, time, argparse, re
from os import system
from os.path import isfile, join
from time import sleep
import multiprocessing as mp
try:
    from armv7l.openvino.inference_engine import IENetwork, IEPlugin
except:
    from openvino.inference_engine import IENetwork, IEPlugin
import heapq
import threading

lastresults = None
threads = []
processes = []
frameBuffer = None
results = None
fps = ""
detectfps = ""
framecount = 0
detectframecount = 0
time1 = 0
time2 = 0
cam = None
camera_width = 320
camera_height = 320
window_name = ""
# ssd_detection_mode = 1
# face_detection_mode = 0
elapsedtime = 0.0

# LABELS = [['background',
#            'aeroplane', 'bicycle', 'bird', 'boat',
#            'bottle', 'bus', 'car', 'cat', 'chair',
#            'cow', 'diningtable', 'dog', 'horse',
#            'motorbike', 'person', 'pottedplant',
#            'sheep', 'sofa', 'train', 'tvmonitor'],
#           ['background', 'face']]

# LABELS = [['background',
#            'aeroplane', 'bicycle', 'bird', 'boat',
#            'bottle', 'bus', 'car', 'cat', 'chair',
#            'cow', 'diningtable', 'dog', 'horse',
#            'motorbike', 'person', 'pottedplant',
#            'sheep', 'sofa', 'train', 'tvmonitor']]

LABELS = [['background','person']]

def camThread(LABELS, results, frameBuffer, camera_width, camera_height, vidfps, number_of_camera):
    global fps
    global detectfps
    global lastresults
    global framecount
    global detectframecount
    global time1
    global time2
    global cam
    global window_name

    cam = cv2.VideoCapture(number_of_camera)
    if cam.isOpened() != True:
        print("USB Camera Open Error!!!")
        sys.exit(0)
    cam.set(cv2.CAP_PROP_FPS, vidfps)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
    window_name = "USB Camera"

    # cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(window_name,cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 640, 640)

    while True:
        t1 = time.perf_counter()

        # 读取摄像头数据，s是布尔值，看读取是否成功，color_image是三维矩阵
        s, color_image = cam.read()
        if not s:
            continue
        if frameBuffer.full():
            frameBuffer.get()
        frames = color_image

        height = color_image.shape[0]
        width = color_image.shape[1]
        #所以frameBuffer中放的是视频图像的引用（猜测）
        frameBuffer.put(color_image.copy())
        res = None
        # 如果有结果的话，那么就画出来这一帧，要不然就用上一帧的结果？
        if not results.empty():
            #非阻塞获得res
            res = results.get(False)
            detectframecount += 1
            imdraw = overlay_on_image(frames, res, LABELS)
            lastresults = res
        else:
            imdraw = overlay_on_image(frames, lastresults, LABELS)

        cv2.imshow(window_name, cv2.resize(imdraw, (width, height)))

        if cv2.waitKey(1)&0xFF == ord('q'):
            sys.exit(0)

        ## Print FPS
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


# l = Search list
# x = Search target value
def searchlist(l, x, notfoundvalue=-1): #这个函数应该可以精简
    if x in l:
        return l.index(x)
    else:
        return notfoundvalue


def async_infer(ncsworker):

    while True:
        ncsworker.predict_async()


class NcsWorker(object):

    def __init__(self, devid, frameBuffer, results, camera_width, camera_height, number_of_ncs):
        self.devid = devid #设备的ID
        self.frameBuffer = frameBuffer

        # self.model_xml = "./lrmodel/MobileNetSSD/MobileNetSSD_deploy.xml"
        # self.model_bin = "./lrmodel/MobileNetSSD/MobileNetSSD_deploy.bin"
        # self.model_xml = "/home/pi/Desktop/ssd_mobilenet_v1_ssd/pedestrian-detection-adas-0002.xml"
        # self.model_bin = "/home/pi/Desktop/ssd_mobilenet_v1_ssd/pedestrian-detection-adas-0002.bin"
        # self.model_xml = "/home/pi/Desktop/ssd_mobilenet_v2_ssd/person-detection-retail-0013.xml"
        # self.model_bin = "/home/pi/Desktop/ssd_mobilenet_v2_ssd/person-detection-retail-0013.bin"
        # self.model_xml = "./lrmodel/MobileNetSSD_pedestrain/MobileNetSSD_deploy_pedestrain.xml"
        # self.model_bin = "./lrmodel/MobileNetSSD_pedestrain/MobileNetSSD_deploy_pedestrain.bin"
        # self.model_xml = "lrmodel/mobilenet_ssd_person/mobilenet_iter_20000.xml"
        # self.model_bin = "lrmodel/mobilenet_ssd_person/mobilenet_iter_20000.bin"
        self.model_xml = "/home/pi/Desktop/mobilenet_v1_ssd_person/frozen_inference_graph.xml"
        self.model_bin = "/home/pi/Desktop/mobilenet_v1_ssd_person/frozen_inference_graph.bin"
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.num_requests = 4 #暂不清楚这个是干什么的
        self.inferred_request = [0] * self.num_requests #[0,0,0,0]
        self.heap_request = []
        self.inferred_cnt = 0 
        self.plugin = IEPlugin(device="MYRIAD")
        self.net = IENetwork(model=self.model_xml, weights=self.model_bin)
        self.input_blob = next(iter(self.net.inputs)) #暂不清楚这个是干什么的
        self.exec_net = self.plugin.load(network=self.net, num_requests=self.num_requests)
        self.results = results
        self.number_of_ncs = number_of_ncs


    def image_preprocessing(self, color_image): #这个可能要改

        # prepimg = cv2.resize(color_image, (672, 384))
        prepimg = cv2.resize(color_image, (300, 300))
        prepimg = prepimg - 127.5
        prepimg = prepimg * 0.007843
        prepimg = prepimg[np.newaxis, :, :, :]     # Batch size axis add
        prepimg = prepimg.transpose((0, 3, 1, 2))  # NHWC to NCHW
        return prepimg


    def predict_async(self):
        try:

            if self.frameBuffer.empty():
                return

            prepimg = self.image_preprocessing(self.frameBuffer.get()) #处理后带有batch_size的图片
            reqnum = searchlist(self.inferred_request, 0)  #返回inferred_request的第一个0的index

            if reqnum > -1:
                self.exec_net.start_async(request_id=reqnum, inputs={self.input_blob: prepimg}) #为指定的推理请求启动异步推理
                self.inferred_request[reqnum] = 1 
                self.inferred_cnt += 1 
                if self.inferred_cnt == sys.maxsize: #超了就重置inferred_request
                    self.inferred_request = [0] * self.num_requests
                    self.heap_request = []
                    self.inferred_cnt = 0
                heapq.heappush(self.heap_request, (self.inferred_cnt, reqnum))

            cnt, dev = heapq.heappop(self.heap_request) #弹出最小值

            if self.exec_net.requests[dev].wait(0) == 0:    # 0-立即返回推理状态，结果available的话直接使用
                self.exec_net.requests[dev].wait(-1)    #那为什么还要等？？
                out = self.exec_net.requests[dev].outputs["detection_out"].flatten() 
                self.results.put([out])
                self.inferred_request[dev] = 0
            else:
                heapq.heappush(self.heap_request, (cnt, dev)) #要不然塞回去

        except:
            import traceback
            traceback.print_exc()


def inferencer(results, frameBuffer,camera_width, camera_height, number_of_ncs):

    # 初始化推理线程
    threads = []
    for devid in range(number_of_ncs):
        thworker = threading.Thread(target=async_infer, args=(NcsWorker(devid, frameBuffer, results, camera_width, camera_height, number_of_ncs),))
        thworker.start()
        threads.append(thworker)
    #让每个MYRAID都处理完
    for th in threads:
        th.join()

#画图像
def overlay_on_image(frames, object_infos, LABELS):

    try:

        color_image = frames
        #如果目标检测的结果
        if isinstance(object_infos, type(None)):
            return color_image

        # 显示图像
        height = color_image.shape[0]
        width = color_image.shape[1]
        entire_pixel = height * width
        img_cp = color_image.copy()

        for (object_info, LABEL) in zip(object_infos, LABELS):

            drawing_initial_flag = True

            for box_index in range(100):
                if object_info[box_index + 1] == 0.0:
                    break
                base_index = box_index * 7
                if (not np.isfinite(object_info[base_index]) or
                    not np.isfinite(object_info[base_index + 1]) or
                    not np.isfinite(object_info[base_index + 2]) or
                    not np.isfinite(object_info[base_index + 3]) or
                    not np.isfinite(object_info[base_index + 4]) or
                    not np.isfinite(object_info[base_index + 5]) or
                    not np.isfinite(object_info[base_index + 6])):
                    continue

                x1 = max(0, int(object_info[base_index + 3] * height))
                y1 = max(0, int(object_info[base_index + 4] * width))
                x2 = min(height, int(object_info[base_index + 5] * height))
                y2 = min(width, int(object_info[base_index + 6] * width))

                object_info_overlay = object_info[base_index:base_index + 7]

                min_score_percent = 60

                source_image_width = width
                source_image_height = height

                base_index = 0
                class_id = object_info_overlay[base_index + 1]
                percentage = int(object_info_overlay[base_index + 2] * 100)
                if (percentage <= min_score_percent):
                    continue

                box_left = int(object_info_overlay[base_index + 3] * source_image_width)
                box_top = int(object_info_overlay[base_index + 4] * source_image_height)
                box_right = int(object_info_overlay[base_index + 5] * source_image_width)
                box_bottom = int(object_info_overlay[base_index + 6] * source_image_height)

                label_text = LABEL[int(class_id)] + " (" + str(percentage) + "%)"

                box_color = (255, 128, 0)
                box_thickness = 1
                cv2.rectangle(img_cp, (box_left, box_top), (box_right, box_bottom), box_color, box_thickness)
                label_background_color = (125, 175, 75)
                label_text_color = (255, 255, 255)
                label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                label_left = box_left
                label_top = box_top - label_size[1]
                if (label_top < 1):
                    label_top = 1
                label_right = label_left + label_size[0]
                label_bottom = label_top + label_size[1]
                cv2.rectangle(img_cp, (label_left - 1, label_top - 1), (label_right + 1, label_bottom + 1), label_background_color, -1)
                cv2.putText(img_cp, label_text, (label_left, label_bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_text_color, 1)


        cv2.putText(img_cp, fps,       (width-170,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (38,0,255), 1, cv2.LINE_AA)
        cv2.putText(img_cp, detectfps, (width-170,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (38,0,255), 1, cv2.LINE_AA)
        return img_cp

    except:
        import traceback
        traceback.print_exc()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-cn','--numberofcamera',dest='number_of_camera',type=int,default=-1,help='USB camera number. (Default=0)')
    parser.add_argument('-wd','--width',dest='camera_width',type=int,default=320,help='Width of the frames in the video stream. (Default=320)')
    parser.add_argument('-ht','--height',dest='camera_height',type=int,default=320,help='Height of the frames in the video stream. (Default=240)')
    # parser.add_argument('-sd','--ssddetection',dest='ssd_detection_mode',type=int,default=1,help='[Future functions] SSDDetectionMode. (0:=Disabled, 1:=Enabled Default=1)')
    # parser.add_argument('-fd','--facedetection',dest='face_detection_mode',type=int,default=0,help='[Future functions] FaceDetectionMode. (0:=Disabled, 1:=Full, 2:=Short Default=0)')
    parser.add_argument('-numncs','--numberofncs',dest='number_of_ncs',type=int,default=1,help='Number of NCS. (Default=1)')
    parser.add_argument('-vidfps','--fpsofvideo',dest='fps_of_video',type=int,default=30,help='FPS of Video. (Default=30)')

    args = parser.parse_args()

    number_of_camera = args.number_of_camera
    camera_widthcamera_width  = args.camera_width
    camera_height = args.camera_height
    # ssd_detection_mode = args.ssd_detection_mode
    # face_detection_mode = args.face_detection_mode
    number_of_ncs = args.number_of_ncs
    vidfps = args.fps_of_video

    # if ssd_detection_mode == 0 and face_detection_mode != 0:
    #     del(LABELS[0])

    try:

        mp.set_start_method('forkserver')
        # 这就是个空队列，应该是共有的东西？
        frameBuffer = mp.Queue(10)
        results = mp.Queue()

        # Start streaming
        p = mp.Process(target=camThread,
                       args=(LABELS, results, frameBuffer, camera_width, camera_height, vidfps, number_of_camera),
                       daemon=True)
        p.start()
        processes.append(p)

        # Start detection MultiStick
        # Activation of inferencer
        p = mp.Process(target=inferencer,
                       args=(results, frameBuffer, camera_width, camera_height, number_of_ncs),
                       daemon=True)
        p.start()
        processes.append(p)

        while True:
            sleep(1)

    except:
        import traceback
        traceback.print_exc()
    finally:
        for p in range(len(processes)):
            processes[p].terminate()

        print("\n\nFinished\n\n")
