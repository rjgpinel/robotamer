from robotamer.envs.base import BaseEnv


class PickEnv(BaseEnv):
    def __init__(self, cam_list, depth=False):
        super(PickEnv, self).__init__(cam_list=cam_list, depth=depth)
