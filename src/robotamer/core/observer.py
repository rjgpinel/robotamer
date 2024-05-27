import os
import tf2_ros
import rospy
import numpy as np
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from robotamer.core.constants import ROBOT_BASE_FRAME
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
        self._info_topic = os.path.join(os.path.dirname(topic), 'camera_info')
        self.info = rospy.wait_for_message(
            self._info_topic, CameraInfo, timeout=None)
        self._im_msg = None
        self._sub = rospy.Subscriber(
            topic, Image, self.save_last_image, queue_size=1, buff_size=2 ** 24
        )

        self.intrinsics = self.info_to_intrinsics()

        deadline = rospy.Time.now() + rospy.Duration(1.0)
        while not rospy.core.is_shutdown() and self._im_msg is None:
            if rospy.Time.now() > deadline:
                rospy.logwarn_throttle(
                    1.0, "Waiting for an image ({})...".format(topic)
                )
            rospy.rostime.wallsleep(0.01)

        if rospy.core.is_shutdown():
            raise rospy.exceptions.ROSInterruptException("rospy shutdown")
        
    def info_to_intrinsics(self):
        msg = self.info
        intrinsics = dict(height=msg.height, width=msg.width, fx=msg.K[0], fy=msg.K[4], ppx=msg.K[2], ppy=msg.K[5], K=np.array(msg.K).reshape(3,3))
        return intrinsics



    def save_last_image(self, msg):
        self._im_msg = msg

    def record_image(self, dtype=np.uint8):
        """Return next received image as numpy array in specified encoding.
        @param timeout: time in seconds
        """
        data = np.frombuffer(self._im_msg.data, dtype=dtype)
        data = data.reshape((self._im_msg.height, self._im_msg.width, -1))
        return data
    
    def record_image_sync(self, timeout=None, dtype=np.uint8):
        msg = rospy.wait_for_message(self._topic, Image, timeout=timeout)

        data = np.frombuffer(msg.data, dtype=dtype)
        data = data.reshape((msg.height, msg.width, -1))

        return data

    
class CameraPose(Camera):
    def __init__(self, topic, camera_frame):
        super().__init__(topic)
        self.tf_recorder = TFRecorder(ROBOT_BASE_FRAME, camera_frame)

    def record_image(self, dtype=np.uint8):
        data = super().record_image(dtype=dtype)
        cam_tf = self.tf_recorder.record_tf()
        
        cam_pos = cam_tf.transform.translation
        cam_pos = [cam_pos.x, cam_pos.y, cam_pos.z]

        cam_rot = cam_tf.transform.rotation
        cam_rot = [cam_rot.x, cam_rot.y, cam_rot.z, cam_rot.w]
        cam_euler = euler_from_quaternion(cam_rot)

        return data, (cam_pos, cam_euler)
    
    def get_pose(self):
        cam_tf = self.tf_recorder.record_tf()
        
        cam_pos = cam_tf.transform.translation
        cam_pos = [cam_pos.x, cam_pos.y, cam_pos.z]

        cam_rot = cam_tf.transform.rotation
        cam_rot = [cam_rot.x, cam_rot.y, cam_rot.z, cam_rot.w]
        cam_euler = euler_from_quaternion(cam_rot)
        return (cam_pos, cam_euler)

class CameraAsyncPose(CameraAsync):
    def __init__(self, topic, camera_frame):
        super().__init__(topic)
        self.tf_recorder = TFRecorder(ROBOT_BASE_FRAME, camera_frame)

    def record_image(self, dtype=np.uint8):
        data = super().record_image(dtype=dtype)
        cam_tf = self.tf_recorder.record_tf()
        
        cam_pos = cam_tf.transform.translation
        cam_pos = [cam_pos.x, cam_pos.y, cam_pos.z]

        cam_rot = cam_tf.transform.rotation
        cam_rot = [cam_rot.x, cam_rot.y, cam_rot.z, cam_rot.w]
        cam_euler = euler_from_quaternion(cam_rot)

        return data, (cam_pos, cam_euler)
    

    def record_image_sync(self, dtype=np.uint8):
        data = super().record_image_sync(dtype=dtype)
        cam_tf = self.tf_recorder.record_tf()
        
        cam_pos = cam_tf.transform.translation
        cam_pos = [cam_pos.x, cam_pos.y, cam_pos.z]

        cam_rot = cam_tf.transform.rotation
        cam_rot = [cam_rot.x, cam_rot.y, cam_rot.z, cam_rot.w]
        cam_euler = euler_from_quaternion(cam_rot)

        return data, (cam_pos, cam_euler)
    
    def get_pose(self):
        cam_tf = self.tf_recorder.record_tf()
        
        cam_pos = cam_tf.transform.translation
        cam_pos = [cam_pos.x, cam_pos.y, cam_pos.z]

        cam_rot = cam_tf.transform.rotation
        cam_rot = [cam_rot.x, cam_rot.y, cam_rot.z, cam_rot.w]
        cam_euler = euler_from_quaternion(cam_rot)
        return (cam_pos, cam_euler)

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
