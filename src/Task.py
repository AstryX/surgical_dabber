#!/usr/bin/env python

import sys
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg

import json
import math
from Predictor import predictImageLabels, findOptimalDestination
from Helper import loadPointCloud


moveit_commander.roscpp_initialize(sys.argv)
rospy.init_node('task1', anonymous=True)
robot = moveit_commander.RobotCommander()

group_name = "ur10"
blue_group = moveit_commander.MoveGroupCommander(group_name)
display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path', moveit_msgs.msg.DisplayTrajectory, queue_size=20)

ros_path = "./src/surgical_dabber/src/"
params_path = ros_path+"params.json"
predict_num = 161
image_size_rows = 250
image_size_cols = 330
im_path = ros_path + "Dataset/Processed/"

with open(params_path) as json_data_file:  
    data = json.load(json_data_file)
    if 'predict_num' in data:
        predict_num = int(data['predict_num'])
    if 'image_size_rows' in data:
        image_size_rows = data['image_size_rows']
    if 'image_size_cols' in data:
        image_size_cols = data['image_size_cols']
    if 'im_path' in data:
        im_path = ros_path + data['im_path']

before_path = im_path+'pc_mock/before.PCD'
after_path = im_path+'pc_mock/after.PCD'

before = loadPointCloud(before_path)
after = loadPointCloud(after_path)
print('Before and after shape after cloud loading:')
print(before.shape)
print(after.shape)

before_z = before['z']
after_z = after['z']

pred_labels, final_contours, loaded_image = predictImageLabels(params_path, predict_num, im_path, ros_path)
top_pool_id, top_pool_deepest_point_id, top_pool_volume = findOptimalDestination(before_z, after_z,
    image_size_rows, image_size_cols, pred_labels, final_contours, loaded_image)
    
max_range_x = 1.0
max_range_y = 1.0
base_image_z = -0.5

step_x = (max_range_x * 2) / float(image_size_cols)
step_y = (max_range_y * 2) / float(image_size_rows)

print('Step x: ' + str(step_x))
print('Step y: ' + str(step_y))

center_x = step_x * (image_size_cols/2)
center_y = step_y * (image_size_rows/2)

print('Center x: ' + str(center_x))
print('Center y: ' + str(center_y))

#TODO Add extra to base z to account for depth
goal_x_idx = top_pool_deepest_point_id % image_size_cols
print('Goal x index: ' + str(goal_x_idx))
goal_x = step_x * goal_x_idx - center_x

goal_y_idx = math.floor(top_pool_deepest_point_id / image_size_cols)
print('Goal y index: ' + str(goal_y_idx))
goal_y = max_range_y - step_y * goal_y_idx

goal_z = base_image_z

print('Robot goal coordinates:')
print('Goal x: ' + str(goal_x))
print('Goal y: ' + str(goal_y))
print('Goal z: ' + str(goal_z))
pose_goal = geometry_msgs.msg.Pose()
pose_goal.orientation.x = -1.0
#pose_goal.position.x = 0.7
#pose_goal.position.y = -0.5
#pose_goal.position.z = -0.5
pose_goal.position.x = goal_x
pose_goal.position.y = goal_y
pose_goal.position.z = goal_z
#pose_goal.position.x = 0
#pose_goal.position.y = 0.35
#pose_goal.position.z = -1.15
blue_group.set_pose_target(pose_goal)


blue_plan = blue_group.go(wait=True)
