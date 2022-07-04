import pinocchio as pin
import numpy as np

from PIL import Image


def blend(im_a, im_b, alpha):
    im_a = Image.fromarray(im_a)
    im_b = Image.fromarray(im_b)
    im_blend = Image.blend(im_a, im_b, alpha).convert("RGB")
    im_blend = np.asanyarray(im_blend).copy()
    return im_blend


def compute_goal_pose(dt, v_xyz, v_rpy, start_pose):
    # delta pose translaion during duration dt
    delta_trans = pin.exp6(np.concatenate((v_xyz, [0, 0, 0])) * dt)
    # delta pose rotation during duration dt
    delta_rot = pin.exp6(np.concatenate(([0, 0, 0], v_rpy)) * dt)
    goal_pose = delta_trans * start_pose * delta_rot
    return goal_pose
