import copy

import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation


def pos_to_hom(pos):
    mat = np.eye(4)
    mat[:3, 3] = pos[:3]
    return mat


def quat_to_hom(quat):
    rot = Rotation.from_quat(quat)
    mat = np.eye(4)
    mat[:3, :3] = rot.as_matrix()
    return mat


def euler_to_hom(euler, standard="xyz"):
    rot = Rotation.from_euler(standard , euler, degrees=False)
    mat = np.eye(4)
    mat[:3, :3] = rot.as_matrix()
    return mat


def hom_to_pos(mat):
    return mat[:3, 3].copy()


def hom_to_quat(mat):
    rot = Rotation.from_matrix(mat[:3, :3])
    quat = rot.as_quat()
    return quat


def pos_quat_to_hom(pos, quat):
    trans_mat = pos_to_hom(pos)
    quat_mat = quat_to_hom(quat)
    return np.matmul(trans_mat, quat_mat)


def pos_euler_to_hom(pos, euler, standard="xyz"):
    trans_mat = pos_to_hom(pos)
    euler_mat = euler_to_hom(euler, standard=standard)
    return np.matmul(trans_mat, euler_mat)


def project(world_pos_quat, view_mat, proj_mat):
    world_T_p = pos_quat_to_hom(*world_pos_quat)
    cam_T_world = view_mat
    cam_T_p = cam_T_world @ world_T_p
    cam_p = cam_T_p[:3, 3]
    cam_p_hom = cam_p / cam_p[2]
    pix_p = proj_mat @ cam_p_hom.T
    pix_p = pix_p[:2].astype(int)
    return pix_p


def depth_to_pcd(depth, intrinsics, depth_scale=1):
    h, w = depth.shape
    fx = intrinsics["fx"]
    fy = intrinsics["fy"]
    cx = intrinsics["ppx"]
    cy = intrinsics["ppy"]
    pcd = np.zeros((h, w, 3))

    pcd[:, :, 0] = np.arange(w).repeat(h).reshape((w, h)).T # u -> x
    pcd[:, :, 1] = np.arange(h).repeat(w).reshape(h, w) # v -> y
    pcd[:, :, 2] = depth / depth_scale


    pcd[:, :, 0] =  (pcd[:, :, 0] - cx) * pcd[:, :, 2] / fx 
    pcd[:, :, 1] =  (pcd[:, :, 1] - cy) * pcd[:, :, 2] / fy

    return pcd


def transform_pcd(pcd, transform_matrix):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(copy.copy(pcd.reshape(-1, 3)))
    pc.transform(transform_matrix)
    pcd = np.asarray(pc.points).reshape(pcd.shape[0], pcd.shape[1], 3)
    return pcd



