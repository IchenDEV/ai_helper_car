from django.http import HttpResponse, StreamingHttpResponse, JsonResponse

# Create your views here.
from aicar.actions.cam_ai import get_obj_pos_stream, face_cmp
from aicar.actions.video_capture import video, encode_image


def index(request):
    return HttpResponse("Hello, world. You're at the polls index.")


def car_cam():
    return HttpResponse("Hello, world. You're at the polls index.")


def car_cam_up():
    return HttpResponse("Hello, world. You're at the polls index.")


def car_cam_down():
    return HttpResponse("Hello, world. You're at the polls index.")


def car_cam_left():
    return HttpResponse("Hello, world. You're at the polls index.")


def car_cam_right():
    return HttpResponse("Hello, world. You're at the polls index.")


def car_cam_center():
    return HttpResponse("Hello, world. You're at the polls index.")


def car_getVideo(request):
    return StreamingHttpResponse(video(), content_type='multipart/x-mixed-replace; boundary=frame')


def car_obj_dect(request):
    return StreamingHttpResponse(get_obj_pos_stream(), content_type='multipart/x-mixed-replace; boundary=frame')


def car_face_cmp(request):
    return JsonResponse(face_cmp())
