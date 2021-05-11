import threading

from django.shortcuts import render
from django.http import HttpResponse

# Create your views here.
from aicar.actions.car_action import move


def car_move(request):
    x = request.POST["x"]
    y = request.POST["y"]
    dis = request.POST["dis"]
    move(x, y, dis)
    return HttpResponse("OK")


def car_stop():
    return HttpResponse("Hello, world. You're at the polls index.")


def car_follow():
    # th1 = threading.Thread(target=follow)
    # th1.setDaemon(True)
    # th1.start()
    return HttpResponse("OK")


def car_stop_follow():
    # stop_follow()
    return HttpResponse("OK")
