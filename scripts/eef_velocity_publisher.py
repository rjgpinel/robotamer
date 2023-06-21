#!/usr/bin/env python

from absl import app
from absl import flags
from geometry_msgs.msg import Vector3
import numpy as np

from robotamer.core.observer import TFRecorder 
from robotamer.core.constants import EEF_FRAME, ROBOT_BASE_FRAME
import rospy


flags.DEFINE_enum('arm', 'left', ['left', 'right'],
                  'The arm for which to publish velocity.')
flags.DEFINE_bool('verbose', False,
                  'Whether to log duration and velocity.')

FLAGS = flags.FLAGS


class VelocityPublisher:

    def __init__(self, arm):
        self.eef_frame = EEF_FRAME[arm]
        self._eef_tf_recorder = TFRecorder(ROBOT_BASE_FRAME, self.eef_frame)
        self.publisher = rospy.Publisher(f'{arm}_eef_velocity', Vector3, queue_size=1)
        self.prev_pos = None
        self.prev_time = None

    def _eef_pos(self):
        tf = self._eef_tf_recorder.record_tf()
        time = tf.header.stamp
        pos = tf.transform.translation
        pos = np.array([pos.x, pos.y, pos.z])
        return pos, time

    def compute_velocity(self):
        pos, time = self._eef_pos()
        if self.prev_pos is not None and self.prev_time is not None:
            dt = (time - self.prev_time).to_sec()
            if dt == 0:
                if FLAGS.verbose:
                    print('Duration', dt)
            else:
                vel = (pos - self.prev_pos) / dt
                self.publisher.publish(*vel)
                if FLAGS.verbose:
                    print('Velocity', vel, 'Dutation', dt)
        self.prev_pos = pos
        self.prev_time = time


def main(_):
    try:
        rospy.init_node('eef_velocity_publisher')
        rate = rospy.Rate(20)
        publisher = VelocityPublisher(FLAGS.arm)
        while not rospy.is_shutdown():
            publisher.compute_velocity()
            rate.sleep()
    except rospy.ROSInterruptException:
        print('Exiting')
    



if __name__ == '__main__':
    app.run(main)
