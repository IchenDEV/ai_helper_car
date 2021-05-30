from django.urls import path

from .views import cam_views, speech_views, move_views

urlpatterns = [
    path('', cam_views.index, name='index'),
    path('video', cam_views.car_getVideo, name='video'),
    path('obj', cam_views.car_obj_dect, name="obj"),

    path('face', cam_views.car_face_cmp, name="face"),
    path('speech', speech_views.car_speech_control, name="speech"),

    path('cam/action/up', cam_views.car_cam_up, name="cam up"),
    path('cam/action/down', cam_views.car_cam_down, name="cam down"),
    path('cam/action/left', cam_views.car_cam_left, name="cam left"),
    path('cam/action/right', cam_views.car_cam_right, name="cam right"),
    path('cam/action/init', cam_views.car_cam_init, name="cam init"),

    path('action/move', move_views.car_move, name="move"),
    path('action/action', move_views.car_action, name="action"),
    path('action/advance', move_views.car_advance, name="advance"),
    path('action/back', move_views.car_back, name="back"),
    path('action/spin/left', move_views.car_spin_left, name="spin left"),
    path('action/spin/right', move_views.car_spin_right, name="spin right"),

    path('action/follow', move_views.car_follow, name="follow"),
    path('action/follow/stop', move_views.car_stop_follow, name="stop follow"),

]
