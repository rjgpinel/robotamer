#!/usr/bin/env python
"""Publisher for dummy joystick messages."""
import rospy
from sensor_msgs.msg import Joy

def talker():
    pub = rospy.Publisher('joy', Joy, queue_size=1)
    rospy.init_node('teleop', anonymous=True)
    rate = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
        cmd = Joy()
        # cmd.axes = [1, 0.5, 0, 0, 0, 0]
        cmd.axes = [0] * 6
        cmd.buttons = [0] * 12
        key = input()
        if key == 'w':
            cmd.axes = [0, 1, 0, 0, 0, 0]
        elif key == 'a':
            cmd.axes = [0.5, 0, 0, 0, 0, 0]
        elif key == 's':
            cmd.axes = [0, -1, 0, 0, 0, 0]
        elif key == 'd':
            cmd.axes = [-0.5, 0, 0, 0, 0, 0]
        elif key == 'x':
            cmd.buttons = [1] + [0] * 11
        rospy.loginfo(cmd)
        pub.publish(cmd)
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
