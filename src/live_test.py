#!/usr/bin/env python

import sys
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from moveit_msgs.msg import RobotState
from sensor_msgs.msg import JointState, Image, PointCloud2
from std_msgs.msg import Header
import visualization_msgs
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from moveit_commander.conversions import pose_to_list
import json
import math
import time
import cv2
import pypcd
from joblib import dump
from cv_bridge import CvBridge
from Predictor import predictImageLabels, findOptimalDestination
from Helper import loadPointCloud
from robotiq_85_msgs.msg import GripperCmd
#from robotiq_85_msgs import GripperCmd

pub = rospy.Publisher('/gripper/cmd', GripperCmd, queue_size=10)
moveit_commander.roscpp_initialize(sys.argv)
rospy.init_node('task1', anonymous=True)
scene = moveit_commander.PlanningSceneInterface()
robot = moveit_commander.RobotCommander()

ros_path = "./src/surgical_dabber/src/"
plan_path = ros_path + 'Plans/'
params_path = ros_path+"params.json"
predict_num = 200
image_size_rows = 250
image_size_cols = 330
bool_display_final_contour = True
im_path = "Dataset/Processed/"

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
    if 'bool_display_final_contour' in data:
        bool_display_final_contour = data['bool_display_final_contour']

data = rospy.wait_for_message('/camera/color/image_rect_color', Image)
bridge = CvBridge()
cv_image = bridge.imgmsg_to_cv2(data, 'passthrough')
'''cv2.imshow('Potato', cv_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv_image = cv2.resize(cv_image,(image_size_cols, image_size_rows))
cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
cv2.imwrite('img (201).jpg', cv_image)'''
pcd_data = rospy.wait_for_message('/camera/depth_registered/points', PointCloud2)
converted_pcd = pypcd.PointCloud.from_msg(pcd_data)
#converted_pcd.save_pcd('test.pcd', compression='binary_compressed')

close_command = GripperCmd()
close_command.emergency_release = False
close_command.emergency_release_dir = 0
close_command.stop = False
close_command.position = 0.0
close_command.speed = 0.0
close_command.force = 0.0

pub.publish(close_command)
close_command.position = 1.0

rospy.sleep(2.0)

pub.publish(close_command)

rospy.sleep(2.0)

quit()

before_path = im_path+'pc_mock/before_'+str(3)+'.PCD'
after_path = im_path+'pc_mock/after_'+str(3)+'.PCD'
before = loadPointCloud(before_path)
after = loadPointCloud(after_path)

before_z = before['z']
after_z = after['z']

pred_labels, final_contours, loaded_image = predictImageLabels(params_path, predict_num, im_path, ros_path)
top_pool_id, top_pool_deepest_point_id, top_pool_volume = findOptimalDestination(before_z, after_z,
                                                                                 image_size_rows, image_size_cols, pred_labels, final_contours, loaded_image, bool_display_final_contour)
print("Run time of Optimal Dab Destination computation: " + str(time.time() - time_benchmark) + " seconds.")
