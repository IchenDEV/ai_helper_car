import time

from aicar.actions.car import get_robot

try:
    '''Part of controling servo    '''
    global leftrightpulse
    leftrightpulse = 1500
    global updownpulse
    updownpulse = 1500


    def cam_up(dis=10):
        global updownpulse
        robot = get_robot()
        updownpulse += dis
        if updownpulse > 2500:
            updownpulse = 2500
        robot.Servo_control(leftrightpulse, updownpulse)


    def cam_down(dis=10):
        global updownpulse
        robot = get_robot()
        updownpulse -= dis
        if updownpulse < 500:
            updownpulse = 500
        robot.Servo_control(leftrightpulse, updownpulse)


    def cam_left(dis=10):
        global leftrightpulse
        robot = get_robot()
        leftrightpulse += dis
        if leftrightpulse > 2500:
            leftrightpulse = 2500
        robot.Servo_control(leftrightpulse, updownpulse)


    def cam_right(dis=10):
        global leftrightpulse
        robot = get_robot()
        leftrightpulse -= dis
        if leftrightpulse < 500:
            leftrightpulse = 500
        robot.Servo_control(leftrightpulse, updownpulse)


    def cam_init():
        global leftrightpulse, updownpulse
        robot = get_robot()
        leftrightpulse = 1500
        updownpulse = 1500
        robot.Servo_control(leftrightpulse, updownpulse)

except:
    pass
