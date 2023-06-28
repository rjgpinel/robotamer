import os
import tf2_ros
import rospy
import numpy as np

from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image
from sensor_msgs.msg import JointState


class Camera:
    def __init__(self, topic):
        self._topic = topic
        self._info_topic = os.path.join(os.path.dirname(topic), 'camera_info')
        self.info = rospy.wait_for_message(
            self._info_topic, CameraInfo, timeout=None)
        self.intrinsics = self.info_to_intrinsics()

    def info_to_intrinsics(self):
        msg = self.info
        intrinsics = dict(height=msg.height, width=msg.width, fx=msg.K[0], fy=msg.K[4], ppx=msg.K[2], ppy=msg.K[5], K=np.array(msg.K).reshape(3,3))
        return intrinsics

    def record_image(self, timeout=None, dtype=np.uint8):
        """Return next received image as numpy array in specified encoding.
        @param timeout: time in seconds
        """
        msg = rospy.wait_for_message(self._topic, Image, timeout=timeout)

        data = np.frombuffer(msg.data, dtype=dtype)
        data = data.reshape((msg.height, msg.width, -1))

        return data


class CameraAsync(Camera):
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

        # self.counter = 0
        # self.times = []

    def save_last_image(self, msg):
        self._im_msg = msg

    def record_image(self, dtype=np.uint8):
        """Return next received image as numpy array in specified encoding.
        @param timeout: time in seconds
        """
        # delay_t = rospy.Time.now() - self._im_msg.header.stamp

        # self.times.append(delay_t.to_sec())
        data = np.frombuffer(self._im_msg.data, dtype=dtype)
        data = data.reshape((self._im_msg.height, self._im_msg.width, -1))

        # self.counter += 1
        # if self.counter % 50 == 0 and self.counter >0:
        #     print(f"Topic: {self._topic} - Mean: {np.mean(self.times)*1000}")
        #     print(f"Topic: {self._topic} - Std: {np.std(self.times)*1000}")
        #     self.times=[]

        return data, self._im_msg.header.stamp.to_sec()


class JointStateRecorder:
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
