from django.urls import path

from .views import cam_views, speech_views, move_views

urlpatterns = [
    path('', cam_views.index, name='index'),
    path('video', cam_views.car_getVideo, name='video'),
    path('obj', cam_views.car_obj_dect, name="obj"),
    path('speech', speech_views.car_speech_control, name="speech"),
    path('follow/stop', move_views.car_stop_follow, name="stop follow"),
    path('move', move_views.move, name="move"),
    path('follow', move_views.car_follow, name="follow"),
    path('face', cam_views.car_face_cmp, name="face"),

]
