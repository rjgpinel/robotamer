from scipy.spatial.transform import Rotation

def safe_random_pose(self):
    current_pose = self.real_robot.robot.left_arm.get_current_pose().pose
    current_pos = [
        current_pose.position.x,
        current_pose.position.y,
        current_pose.position.z,
    ]
    current_ori = [
        current_pose.orientation.x,
        current_pose.orientation.y,
        current_pose.orientation.z,
        current_pose.orientation.w,
    ]
    ori = [pi, 0, pi / 2]

    random_x, random_y, random_z = self.sample_random_gripper_pos()

    waypoints = [
        current_pose,
        make_pose(
            [current_pos[0], current_pos[1], self.safe_height[1]],
            current_ori,
            frame_id="prl_ur5_base",
        ).pose,
        make_pose(
            [random_x, random_y, random_z],
            ori,
            frame_id="prl_ur5_base",
        ).pose,
    ]

    # Move in cartesian space through the waypoints
    path, fraction = self.real_robot.robot.left_arm.compute_cartesian_path(
        waypoints=waypoints,
        eef_step=0.001,
        jump_threshold=0.0,
    )

    if fraction < 0.1:
        rospy.logerr(f"Failed to plan cartesian path. Fraction Path: {fraction}")
        return False

    state = self.real_robot.robot.get_current_state()
    path = self.real_robot.robot.left_arm.retime_trajectory(
        state, path, 0.5, 0.5, "time_optimal_trajectory_generation"
    )

    success = self.real_robot.robot.left_arm.execute(path, wait=True)

    return success


def euler_to_quat(euler, degrees):
    rotation = Rotation.from_euler("xyz", euler, degrees=degrees)
    return rotation.as_quat()


def quat_to_euler(quat, degrees):
    rotation = Rotation.from_quat(quat)
    return rotation.as_euler("xyz", degrees=degrees)

