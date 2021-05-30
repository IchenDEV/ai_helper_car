import threading
import time

# from aicar.actions.car_action import move_advance, move_back
from aicar.utils.api import speechClient
from aicar.utils.voice.voice import voice


def start():
    th1 = threading.Thread(target=get_voice)
    th1.setDaemon(True)
    th1.start()


def play_speech(word):
    result = speechClient.synthesis(word, 'zh', 1, {'vol': 5, })
    print(result)
    if not isinstance(result, dict):
        print("ko")
        with open('./temp/audio.mp3', 'wb') as f:
            f.write(result)
        print("ok")
        import subprocess
        player = subprocess.Popen(["mplayer", "./temp/audio.mp3"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE)


def func(word):
    if "前进" in word:
        print("前进中")
        play_speech("前进中")
        dis = 100
        # move_advance(dis)
    if "后退" in word:
        print("后退中")
        play_speech("后退中")
        dis = 100
        # move_back(dis)
    if "右转" in word:
        print("右转中")
        play_speech("右转中")
        dis = 100
        # move_back(dis)
    if "左转" in word:
        print("左转中")
        play_speech("左转中")
        dis = 100
        # move_back(dis)
    if "抬头" in word:
        print("抬头中")
        play_speech("抬头中")
        dis = 100
        # move_back(dis)
    if "低头" in word:
        print("低头中")
        play_speech("低头中")
        dis = 100
        # move_back(dis)

    if "低头" in word:
        print("低头中")
        play_speech("低头中")
        dis = 100
        # move_back(dis)


def get_voice():
    while True:
        try:
            word = voice()
            func(word)
        except:
            pass
        time.sleep(0.5)


start()
i = 1
while True:
    i = i + 1
