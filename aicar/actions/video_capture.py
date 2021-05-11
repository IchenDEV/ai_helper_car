import cv2
import time
import threading


def Camera_isOpened():
    global stream, cap
    cap = cv2.VideoCapture(stream)


c = 80
width, height = 640, 480
resolution = str(width) + "x" + str(height)
orgFrame = None
Running = True
ret = False
stream = 0  # 摄像头

try:
    Camera_isOpened()
    cap = cv2.VideoCapture(stream)
    cap.set(3, width)
    cap.set(4, height)
except:
    print('Unable to detect camera! \n')


def getOrgFrame():
    global orgFrame
    return orgFrame


def get_image():
    global orgFrame
    global ret
    global Running
    global stream, cap
    global width, height
    while True:
        if Running:
            try:
                if cap.isOpened():
                    ret, orgFrame = cap.read()
                else:
                    time.sleep(0.01)
            except:
                cap = cv2.VideoCapture(stream)
                cap.set(cv2.CV_CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CV_CAP_PROP_FRAME_HEIGHT, height)
        else:
            time.sleep(0.01)


def video():
    global orgFrame
    while True:
        if Running:
            try:
                if cap.isOpened():
                    _, encodedImage = cv2.imencode(".jpg", orgFrame)
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + encodedImage.tobytes() + b'\r\n')
                else:
                    time.sleep(0.01)
            except:
                time.sleep(0.01)
                print("Exception")
        else:
            time.sleep(0.01)


def encode_image():
    global orgFrame
    while True:
        if Running:
            try:
                if cap.isOpened():
                    _, encodedImage = cv2.imencode(".jpg", orgFrame)
                    return encodedImage.tobytes()
                else:
                    time.sleep(0.01)
            except:
                time.sleep(0.01)
                print("Exception")
        else:
            time.sleep(0.01)


th1 = threading.Thread(target=get_image)
th1.setDaemon(True)
th1.start()
