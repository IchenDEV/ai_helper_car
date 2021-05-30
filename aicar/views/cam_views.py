from django.http import HttpResponse, StreamingHttpResponse, JsonResponse

# Create your views here.
from aicar.actions.cam_ai import get_obj_pos_stream, face_cmp
from aicar.actions.car_cam_action import cam_init, cam_up, cam_down, cam_left, cam_right
from aicar.actions.video_capture import video, encode_image


def car_cam_init(request):
    cam_init()
    return HttpResponse("OK")


def car_cam_up(request):
    dis = request.POST["dis"]
    cam_up(dis)
    return HttpResponse("OK")


def car_cam_down(request):
    dis = request.POST["dis"]
    cam_down(dis)
    return HttpResponse("OK")


def car_cam_left(request):
    dis = request.POST["dis"]
    cam_left(dis)
    return HttpResponse("OK")


def car_cam_right(request):
    dis = request.POST["dis"]
    cam_right(dis)
    return HttpResponse("OK")


def car_getVideo(request):
    return StreamingHttpResponse(video(), content_type='multipart/x-mixed-replace; boundary=frame')


def car_obj_dect(request):
    return StreamingHttpResponse(get_obj_pos_stream(), content_type='multipart/x-mixed-replace; boundary=frame')


def car_face_cmp(request):
    return JsonResponse(face_cmp())
