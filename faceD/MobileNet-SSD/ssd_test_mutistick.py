import sys
if sys.version_info.major < 3 or sys.version_info.minor < 4:
    print("Please using python3.4 or greater!")
    sys.exit(1)
import numpy as np
import cv2, io, time, argparse, re
import os
# from os import system
from os.path import  join
from time import sleep
import multiprocessing as mp
try:
    from armv7l.openvino.inference_engine import IENetwork, IEPlugin
except:
    from openvino.inference_engine import IENetwork, IEPlugin
import heapq
import threading
import logging as log
import shutil


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
# camera_width = 320
# camera_height = 320
window_name = ""

elapsedtime = 0.0

LABELS = ['background','person']

def camThread(LABELS, results, frameBuffer, camera_width, camera_height,prob_threshold,predicted_dir_path):
    global fps
    global detectfps
    global lastresults
    global framecount
    global detectframecount
    global time1
    global time2
    global cam
    global window_name

    with open("/home/pi/Desktop/MobileNet-SSD/VOCdevkit/2007_test.txt", 'r') as annotation_file:
        for num, line in enumerate(annotation_file):
            annotation = line.strip().split()
            image_path = annotation[0]
            image_name = image_path.split('/')[-1]
            color_image = cv2.imread(image_path)
            height, width = color_image.shape[0:2]
            # if(num % 100 == 0):
            #     print('num:%s'%num)
            frames = color_image

        #所以frameBuffer中放的是视频图像的引用（猜测）
            frameBuffer.put(color_image.copy())
            res = None
        # 如果有结果的话，那么就画出来这一帧，要不然就用上一帧的结果？
            if not results.empty():
                #非阻塞获得res
                res = results.get(False)
                # print("res shape:", res.shape)
                overlay_on_image(frames, res, LABELS,prob_threshold,num,predicted_dir_path)
                lastresults = res
            else:
                overlay_on_image(frames, lastresults, LABELS,prob_threshold,num,predicted_dir_path)

def async_infer(ncsworker):
    while True:
        ncsworker.predict_async()


class NcsWorker(object):

    def __init__(self, devid, frameBuffer, results, camera_width, camera_height, number_of_ncs,model_xml,model_bin):
        self.devid = devid #设备的ID
        self.frameBuffer = frameBuffer
        log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
        self.model_xml = model_xml
        self.model_bin = model_bin
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.num_requests = 4 #暂不清楚这个是干什么的
        self.inferred_request = [0] * self.num_requests #[0,0,0,0]
        self.heap_request = []
        self.inferred_cnt = 0 
        log.info("Creating Inference Engine...")
        self.plugin = IEPlugin(device="MYRIAD")
        log.info("Loading network files:\n\t{}\n\t{}".format(self.model_xml, self.model_bin))
        self.net = IENetwork(model=self.model_xml, weights=self.model_bin)
        self.input_blob = next(iter(self.net.inputs)) #暂不清楚这个是干什么的
        log.info("Loading IR to the plugin...")
        self.exec_net = self.plugin.load(network=self.net, num_requests=self.num_requests)
        self.results = results
        self.number_of_ncs = number_of_ncs
        print("To close the application, press 'CTRL+C' or q ")

    def image_preprocessing(self, color_image): #这个可能要改
        prepimg = cv2.resize(color_image, (self.camera_height, self.camera_width))
        prepimg = prepimg[np.newaxis, :, :, :]     # Batch size axis add
        prepimg = prepimg.transpose((0, 3, 1, 2))  # NHWC to NCHW
        return prepimg


    def predict_async(self):
        try:

            if self.frameBuffer.empty():
                return

            prepimg = self.image_preprocessing(self.frameBuffer.get()) #处理后带有batch_size的图片
            reqnum=-1
            if 0 in self.inferred_request:
                reqnum=self.inferred_request.index(0)
            # reqnum = searchlist(self.inferred_request, 0)  #返回inferred_request的第一个0的index
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
                out = self.exec_net.requests[dev].outputs["DetectionOutput"]
                self.results.put(out)
                self.inferred_request[dev] = 0
            else:
                heapq.heappush(self.heap_request, (cnt, dev)) #要不然塞回去

        except:
            import traceback
            traceback.print_exc()


def inference(results, frameBuffer,camera_width, camera_height, number_of_ncs,model_xml,model_bin):
    # 初始化推理线程
    threads = []
    for devid in range(number_of_ncs):
        thworker = threading.Thread(target=async_infer, args=(NcsWorker(devid, frameBuffer, results, camera_width, camera_height, number_of_ncs,model_xml,model_bin),))
        thworker.start()
        threads.append(thworker)
    #让每个MYRAID都处理完
    for th in threads:
        th.join()

# 画图像
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

def overlay_on_image(frames, object_infos, LABELS,prob_threshold,num,predicted_dir_path):
    try:
        color_image = frames
        print('num:%s'%num)
        #如果目标检测的结果
        if isinstance(object_infos, type(None)):
            return color_image
        frame_h = color_image.shape[0]
        frame_w = color_image.shape[1]
        frame = color_image.copy()
        predict_result_path = os.path.join(predicted_dir_path, str(num) + '.txt')
        objects = []
        for obj in object_infos[0][0]:
            # 仅当概率大于指定阈值时保存对象
            if obj[2] > prob_threshold:
                objects.append(obj)
        objlen = len(objects)
        for i in range(objlen):
                for j in range(i + 1, objlen):
                    iou = IntersectionOverUnion(objects[i], objects[j]) 
                    if (iou >= 0.5):
                        if objects[i][2] < objects[j][2]:
                            objects[i], objects[j] = objects[j], objects[i]
                        objects[j][2] = 0.0
        with open(predict_result_path, 'w') as f:
            for obj in objects:
                # 仅当概率大于指定阈值时绘制对象
                if obj[2] > prob_threshold:
                    xmin = int(obj[3] * frame_w)
                    ymin = int(obj[4] * frame_h)
                    xmax = int(obj[5] * frame_w)
                    ymax = int(obj[6] * frame_h)
                    class_id = int(obj[1])
                    score = '%.4f' % obj[2]
                    list1=[class_id, score, xmin, ymin, xmax, ymax]
                    bbox_mess = ' '.join('%s' %id for id in list1) + '\n'
                    f.write(bbox_mess)
        return frame
    except:
        import traceback
        traceback.print_exc()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS, help='Show this help message and exit.')
    parser.add_argument("-m", "--model", help="Required. Path to an .xml file with a trained model.",
                      required=True, type=str)
    parser.add_argument('-wd','--width',dest='camera_width',type=int,default=320,help='Width of the frames in the video stream. (Default=320)')
    parser.add_argument('-ht','--height',dest='camera_height',type=int,default=320,help='Height of the frames in the video stream. (Default=320)')
    parser.add_argument('-numncs','--numberofncs',dest='number_of_ncs',type=int,default=1,help='Number of NCS. (Default=1)')
    parser.add_argument("-pt", "--prob_threshold", help="Optional. Probability threshold for detections filtering",
                      default=0.4, type=float)
    args = parser.parse_args()
    

    camera_width  = args.camera_width
    camera_height = args.camera_height
    number_of_ncs = args.number_of_ncs
    prob_threshold =args.prob_threshold
    model_xml = args.model
    model_bin = os.path.splitext(model_xml)[0] + ".bin"
    file_name=os.path.splitext(model_xml)[0].split('/')[-2]
    predicted_dir_path = 'mAP/'+file_name+'_multi_ncs'+str(number_of_ncs)+'/predicted'
    if os.path.exists(predicted_dir_path): shutil.rmtree(predicted_dir_path) #递归删除所有文件
    os.makedirs(predicted_dir_path)
    try:

        mp.set_start_method('forkserver')
        # 这就是个空队列，应该是共有的东西？
        frameBuffer = mp.Queue(10)
        results = mp.Queue()
        # 开始视频流


        # 开始多棒进行处理，激活推理
        p = mp.Process(target=inference,
                       args=(results, frameBuffer, camera_width, camera_height, number_of_ncs,model_xml,model_bin),
                       daemon=True)
        p.start()
        processes.append(p)
        time.sleep(30)
        p = mp.Process(target=camThread,
                args=(LABELS, results, frameBuffer, camera_width, camera_height,prob_threshold,predicted_dir_path),
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
