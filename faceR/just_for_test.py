import threading
import faceR.detector_recognize_openvino_video as drov
import cv2
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    ret, orgFrame = cap.read()
    drov.start_face_detector(orgFrame)
