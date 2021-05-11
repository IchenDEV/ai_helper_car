import threading

IsWatchingTof = True
IsWatchingDHT11 = True
IsWatchingSmoke = True


def init():
    th1 = threading.Thread(target=watch_Tof)
    th1.setDaemon(True)
    th1.start()

    th2 = threading.Thread(target=watch_DHT11)
    th2.setDaemon(True)
    th2.start()

    th3 = threading.Thread(target=watch_Smoke)
    th3.setDaemon(True)
    th3.start()


def watch_Tof():
    while IsWatchingTof:
        a = 1


def watch_DHT11():
    while IsWatchingDHT11:
        a = 1


def watch_Smoke():
    while IsWatchingSmoke:
        a = 1
