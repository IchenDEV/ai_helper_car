from django.shortcuts import render
from django.http import HttpResponse
from aicar.actions.cam_ai import play_speech


def car_play_sound(request):
    return HttpResponse("Hello, world. You're at the polls index.")


def car_speech_control(request):
    play_speech(request.POST["word"])
    return HttpResponse("OK")
