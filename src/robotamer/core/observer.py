import tf2_ros
import rospy
import numpy as np

from sensor_msgs.msg import JointState
from sensor_msgs.msg import Image


class Camera:
    def __init__(self, topic):
        self._topic = topic

    def record_image(self, timeout=None, dtype=np.uint8):
        """Return next received image as numpy array in specified encoding.
        @param timeout: time in seconds
        """
        msg = rospy.wait_for_message(self._topic, Image, timeout=timeout)

        data = np.frombuffer(msg.data, dtype=dtype)
        data = data.reshape((msg.height, msg.width, -1))

        return data


class CameraAsync:
    def __init__(self, topic):
        self._topic = topic
        self._im_msg = None
        self._sub = rospy.Subscriber(
            topic, Image, self.save_last_image, queue_size=1, buff_size=2 ** 24
        )

        deadline = rospy.Time.now() + rospy.Duration(1.0)
        while not rospy.core.is_shutdown() and self._im_msg is None:
            if rospy.Time.now() > deadline:
                rospy.logwarn_throttle(
                    1.0, "Waiting for an image ({})...".format(topic)
                )
            rospy.rostime.wallsleep(0.01)

        if rospy.core.is_shutdown():
            raise rospy.exceptions.ROSInterruptException("rospy shutdown")

    def save_last_image(self, msg):
        self._im_msg = msg

    def record_image(self, dtype=np.uint8):
        """Return next received image as numpy array in specified encoding.
        @param timeout: time in seconds
        """
        data = np.frombuffer(self._im_msg.data, dtype=dtype)
        data = data.reshape((self._im_msg.height, self._im_msg.width, -1))
        return data


class JoinStateRecorder:
    def __init__(self, topic="/joint_states"):
        self._topic = topic

    def record_state(self, timeout=None):
        msg = rospy.wait_for_message(self._topic, JointState, timeout=timeout)

        data = {
            "joint_position": msg.position,
            "joint_names": msg.name,
            "joint_velocity": msg.velocity,
        }
        return data


class TFRecorder:
    def __init__(self, source_frame, target_frame):
        self._source_frame = source_frame
        self._target_frame = target_frame

        self.tf_buffer = tf2_ros.Buffer()
        self.tf2_listener = tf2_ros.TransformListener(self.tf_buffer)

    def record_tf(self, timeout=4.0, now=False):
        now = rospy.Time.now() if now else rospy.Time(0)
        transform = self.tf_buffer.lookup_transform(
            self._source_frame, self._target_frame, now, rospy.Duration(timeout)
        )

        return transform
