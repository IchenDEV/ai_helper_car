import threading

from django.shortcuts import render
from django.http import HttpResponse

# Create your views here.
from aicar.actions.car_action import move, move_back, action_spin_left, action_spin_right, move_advance, action


def car_move(request):
    x = request.POST["x"]
    y = request.POST["y"]
    dis = request.POST["dis"]
    move(x, y, dis)
    return HttpResponse("OK")


def car_action(request):
    w1 = request.POST["w1"]
    w2 = request.POST["w2"]
    w3 = request.POST["w3"]
    w4 = request.POST["w4"]
    dis = request.POST["dis"]
    action(w1, w2, w3, w4, dis)
    return HttpResponse("OK")


def car_advance(request):
    dis = request.POST["dis"]
    move_advance(dis)
    return HttpResponse("OK")


def car_back(request):
    dis = request.POST["dis"]
    move_back(dis)
    return HttpResponse("OK")


def car_spin_left(request):
    dis = request.POST["dis"]
    action_spin_left(dis)
    return HttpResponse("OK")


def car_spin_right(request):
    dis = request.POST["dis"]
    action_spin_right(dis)
    return HttpResponse("OK")


def car_follow():
    # th1 = threading.Thread(target=follow)
    # th1.setDaemon(True)
    # th1.start()
    return HttpResponse("OK")


def car_stop_follow():
    # stop_follow()
    return HttpResponse("OK")
