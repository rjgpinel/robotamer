import numpy as np
import trimesh


files = {
    "flex_finger":'/home/rgarciap/catkin_ws/src/onrobot_ros/onrobot_description/meshes/rg6_v2/visual/flex_finger.stl'
}


for link, filename in files.items():
    model = trimesh.load(filename)
    import pudb; pudb.set_trace()
    print(link, model.bounding_box.vertices())
