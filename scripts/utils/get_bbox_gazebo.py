import sys
import rospy
import rosservice
import geometry_msgs.msg
import gazebo_msgs.msg
import math

from std_msgs.msg import String
from gazebo_msgs.srv import GetModelState

# Initialize node
rospy.init_node('get_bounding_box')
model_state_getter = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState, persistent=True)

model_name = 'prl_ur5_robot::prl_ur5_base'  # Replace with your model name
state_object = model_state_getter(model_name, 'world')

import pudb; pudb.set_trace()
# Get model dimensions
dimensions = state_object.wrench.force
width = dimensions.x
depth = dimensions.y
height = dimensions.z

# Calculate bounding box corners
half_width = width / 2
half_depth = depth / 2
half_height = height / 2

# Corners of the bounding box
corners = [
    (-half_width, -half_depth, half_height),  # Bottom left front
    (half_width, -half_depth, half_height),   # Bottom right front
    (half_width, half_depth, half_height),    # Top right front
    (-half_width, half_depth, half_height),   # Top left front
]

# Print corners
for i, corner in enumerate(corners):
    print(f"Corner {i+1}: {corner}")