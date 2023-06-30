import pinocchio as pin
import numpy as np

from datetime import datetime
from PIL import Image

from scipy.spatial.transform import Rotation
from torchvision.transforms.functional import resize
from torchvision.transforms import InterpolationMode
from torchvision.transforms import Resize

def get_timestamp():
    return datetime.now().strftime('%Y-%m-%dT%H-%M-%S')


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


def crop_center(im, crop_h, crop_w):
    h, w = im.shape[-2], im.shape[-1]
    start_x = w//2 - (crop_w//2)
    start_y = h//2 - (crop_h//2)    
    return im[..., start_y:start_y+crop_w,start_x:start_x+crop_h], start_x, start_y


def resize(im, new_size, im_type="rgb"):
    if im_type == "rgb":
        interpolation = InterpolationMode.BILINEAR
    elif im_type == "depth":
        interpolation = InterpolationMode.NEAREST
    elif im_type == "gripper_attn":
        interpolation = InterpolationMode.NEAREST
    elif im_type == "pc":
        interpolation = InterpolationMode.NEAREST
    
    
    orig_h, orig_w = im.shape[-2], im.shape[-1]
    if orig_h < orig_w:
        ratio = (new_size/orig_h)
    else:
        ratio = (new_size/orig_w)
    
    h_resize = int(orig_h * ratio)
    w_resize = int(orig_w * ratio)

    resize = Resize((h_resize, w_resize), interpolation=interpolation)

    new_im = resize(im)
    return new_im, ratio


