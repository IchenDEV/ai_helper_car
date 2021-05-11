import time

try:
    from Raspblock import Raspblock

    from aicar.actions.cam_ai import get_obj_pos


    def action(w1, w2, w3, w4, dis):
        robot = Raspblock()
        while dis > 0:
            dis -= 1
            robot.Speed_Wheel_control(w1, w2, w3, w4)
            time.sleep(0.5)
        del robot


    def move(x, y, dis):
        robot = Raspblock()
        while dis > 0:
            dis -= 1
            robot.Speed_axis_Yawhold_control(x, y)  # Advance
            time.sleep(0.5)
        del robot


    Following = False


    def follow():
        global Following
        Following = True
        while Following:
            pos = get_obj_pos()
            right = 640 - pos.width - pos.left
            move(right - pos.left, 5, 2)


    def stop_follow():
        global Following
        Following = False

except:
    pass
