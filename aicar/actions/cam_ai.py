import base64

from aicar.actions.video_capture import encode_image, getOrgFrame
from aicar.utils.api import imageClassifyClient, speechClient, faceClient
from playsound import playsound

from faceR import faceR


def get_obj_pos_stream():
    index = 0
    while True:
        index += 1
        encodedImage = encode_image()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + encodedImage + b'\r\n')
        if index % 100 == 0:
            res = imageClassifyClient.objectDetect(encodedImage)
            print(res)


def get_obj_pos():
    encodedImage = encode_image()
    return imageClassifyClient.objectDetect(encodedImage)


def play_speech(word):
    result = speechClient.synthesis(word, 'zh', 1, {'vol': 5, })
    if not isinstance(result, dict):
        with open('./temp/audio.mp3', 'wb') as f:
            f.write(result)
        import subprocess
        player = subprocess.Popen(["mplayer", "./temp/audio.mp3"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def upload_face(image):
    with open('./temp/face/face.png', 'wb') as f:
        f.write(image)


def face_cmp():
    image = encode_image()

    res = faceClient.match([
        {
            'image': str(base64.b64encode(open("./temp/face/face.png", "rb").read()))[2:],
            'image_type': 'BASE64',
        },
        {
            'image': str(base64.b64encode(image))[2:],
            'image_type': 'BASE64',
        }
    ])
    print(res)
    return res


def face_r():
    image = getOrgFrame()

    return faceR(image)
