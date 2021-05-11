from aip import AipFace, AipSpeech, AipImageClassify
import base64

""" 你的 APPID AK SK """
APP_ID = '24027772'
API_KEY = 'V7vP7ZX8ZuUqQmwawQagdCr1'
SECRET_KEY = 'qE3rfHjABvKv6CZlAHor18yybeOS8VLq'

imageClassifyClient = AipImageClassify(APP_ID, API_KEY, SECRET_KEY)
faceClient = AipFace(APP_ID, API_KEY, SECRET_KEY)
speechClient = AipSpeech(APP_ID, API_KEY, SECRET_KEY)

