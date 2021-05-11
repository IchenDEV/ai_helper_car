from django.shortcuts import render
from django.http import HttpResponse
# Create your views here.
def car_smoke(request):
    return HttpResponse("Hello, world. You're at the polls index.")

def car_dht11(request):
    return HttpResponse("Hello, world. You're at the polls index.")

def car_tof(request):
    return HttpResponse("Hello, world. You're at the polls index.")