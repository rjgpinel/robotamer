import rospy

import numpy as np

from sensor_msgs.msg import Joy


class ActionRepeater:

    def __init__(self):
        self.prev_action_zero = True
        self.publisher = rospy.Publisher('joy_teleop', Joy, queue_size=1)

    def callback(self, action):
        """Only send action if non-zero or if the last action was nonzero."""
        zero_action = not np.any(action.axes) and not np.any(action.buttons)
        if not zero_action or not self.prev_action_zero:
            self.publisher.publish(action)
        self.prev_action_zero = not np.any(action.axes)


def main():
    rospy.init_node('teleop_repeater')
    repeater = ActionRepeater()
    try:
        rospy.Subscriber('joy', Joy, repeater.callback, queue_size=1)
        print('Ready to repeat joystick controls')

        rospy.spin()
    except rospy.ROSInterruptException:
        dataset.save()


if __name__ == "__main__":
    main()


