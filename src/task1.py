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
from joblib import dump,load
import pyassimp
import json
import math
import time
import cv2
import pypcd
import tf2_ros
import tf2_py as tf2
import numpy as np
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
from cv_bridge import CvBridge
from Predictor import predictImageLabels, findOptimalDestination
from Helper import loadPointCloud
from robotiq_85_msgs.msg import GripperCmd

def insert_box(x, y, z, scale_x, scale_y, scale_z, cur_scene, obs_name, frame_id):
    box_pose = geometry_msgs.msg.PoseStamped()
    box_pose.header.frame_id = frame_id
    box_pose.pose.position.x = x
    box_pose.pose.position.y = y
    box_pose.pose.position.z = z
    box_pose.pose.orientation.w = 1.0
    box_name = obs_name
    cur_scene.add_box(box_name, box_pose, size=(scale_x, scale_y, scale_z))    
    
def insert_mesh(x, y, z, scale_x, scale_y, scale_z, cur_scene, mesh_name, frame_id, mesh_path):
    box_pose = geometry_msgs.msg.PoseStamped()
    box_pose.header.frame_id = frame_id
    box_pose.pose.position.x = x
    box_pose.pose.position.y = y
    box_pose.pose.position.z = z
    box_pose.pose.orientation.w = 1.0
    box_name = mesh_name
    cur_scene.add_mesh(box_name, box_pose, mesh_path, size=(scale_x, scale_y, scale_z))    
    
def create_mesh_marker(x, y, z, scale_x, scale_y, scale_z, frame_id, mesh_path):
    marker_array = MarkerArray()
    marker = Marker()
    marker.type = marker.MESH_RESOURCE
    marker.mesh_resource = mesh_path
    marker.header.frame_id = frame_id
    marker.action = 0
    marker.header.stamp = rospy.get_rostime()
    marker.id = 0
    marker.pose.orientation.w = 1.0
    marker.scale.x = scale_x
    marker.scale.y = scale_y
    marker.scale.z = scale_z
    marker.mesh_use_embedded_materials = True
    marker.pose.position.x = x
    marker.pose.position.y = y
    marker.pose.position.z = z

    marker_array.markers.append(marker)
    return marker_array
    
def plan_to_goal(x, y, z, cur_group, num_tries, backup_plan):
    full_plan = None
    for i in range(num_tries):
        pose_goal = geometry_msgs.msg.Pose()
        #pose_goal.orientation.x = -1.0
        pose_goal.orientation.w = 1.0
        pose_goal.position.x = x
        pose_goal.position.y = y
        pose_goal.position.z = z
        cur_group.set_pose_target(pose_goal)
        plan = cur_group.plan()

        if plan[0] == True:
            print('Path found after ' + str(i+1) + ' out of ' + str(num_tries) + ' attempts!')
            full_plan = plan
            break

    if ((full_plan is None) and (backup_plan is not None)):
        print('Path not found after ' + str(num_tries) + ' tries! Reverting to the backup choice...')
        full_plan = load(backup_plan)

    return full_plan, pose_goal

def plan_to_goal_joints(joints, cur_group, num_tries, backup_plan):
    full_plan = None
    for i in range(num_tries):
        cur_group.set_joint_value_target(joints)
        plan = cur_group.plan()

        if plan[0] == True:
            print('Path found after ' + str(i+1) + ' out of ' + str(num_tries) + ' attempts!')
            full_plan = plan
            break

    if ((full_plan is None) and (backup_plan is not None)):
        print('Path not found after ' + str(num_tries) + ' tries! Reverting to the backup choice...')
        full_plan = load(backup_plan)

    return full_plan, None
    
def check_if_goal_reached(pose_goal, cur_group, tolerance):
    pose_goal_arr = pose_to_list(pose_goal)
    cur_pose = pose_to_list(cur_group.get_current_pose().pose)

    bool_reached_goal = True
    for i in range(len(cur_pose)):
        if i == 3: #weird glitch where 4th joint has diff result even when it shouldn't
            continue
        pose_diff = abs(cur_pose[i] - pose_goal_arr[i])
        if (pose_diff > tolerance):
            bool_reached_goal = False

    print(pose_goal_arr)
    print(cur_pose)
    print('Did robot reach goal within tolerance: ' + str(bool_reached_goal))
    return bool_reached_goal
    
def reset_start_state(blue_group, active_joints):
    pose_goal_joints = blue_group.get_current_joint_values()
    print('Cur joints')
    print(pose_goal_joints)

    joint_state = JointState()
    joint_state.header = Header()
    joint_state.header.stamp = rospy.Time.now()
    joint_state.name = active_joints
    joint_state.position = pose_goal_joints
    moveit_robot_state = RobotState()
    moveit_robot_state.joint_state = joint_state
    blue_group.set_start_state(moveit_robot_state) 
    
def plan_and_move(location_array, blue_group, ros_path, backup_plan, num_tries, active_joints, x_offset, precomputed_safe_plan=None):
    rospy.sleep(1)
    reset_start_state(blue_group, active_joints)
    combined_path = None
    plan_pose = None
    plan = None
    if precomputed_safe_plan is None:
        if backup_plan is not None:
            combined_path = ros_path+backup_plan
        if len(location_array) == 3:
            plan, plan_pose = plan_to_goal(location_array[0]+x_offset, location_array[1], location_array[2], blue_group, num_tries, combined_path)
        else:
            plan, plan_pose = plan_to_goal_joints(location_array, blue_group, num_tries, combined_path)
        #Saving plans code:
        #if (plan is not None) and (backup_plan is not None):
        #    dump(plan, backup_plan)
        blue_group.execute(plan[1], wait=True)
    else:
        print('Executing precomputed safe plan!')
        blue_group.execute(precomputed_safe_plan, wait=True)
    blue_group.stop()
    blue_group.clear_pose_targets()

    return plan_pose, plan

def cloud_to_map_coordinates(top_pool_deepest_point_id, before):
    optimal_point = before[top_pool_deepest_point_id]
    print('Robot goal coordinates:')
    print('Goal x: ' + str(optimal_point['x']))
    print('Goal y: ' + str(optimal_point['y']))
    print('Goal z: ' + str(optimal_point['z']))
    dabbing_goal = [float(optimal_point['x']), float(optimal_point['y']), float(optimal_point['z'])]
    return dabbing_goal

def pixel_to_map_coordinates(top_pool_deepest_point_id, image_size_cols, image_size_rows):
    max_range_x = 0.33
    max_range_y = max_range_x * image_size_rows/image_size_cols
    base_image_z = 1.1

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

    dabbing_goal = [goal_x, goal_y, goal_z]
    return dabbing_goal
    
def extract_optimal_goal(before_path, after_path, predict_num, image_size_cols, image_size_rows,
                        im_path, params_path, ros_path, bool_display_final_contour, bool_should_simulate_goal,
                        clf, dim_red_model, scaler):
    before = before_path
    after = after_path
    if bool_should_simulate_goal is True:
        before = loadPointCloud(before_path)
        after = loadPointCloud(after_path)
    else:
        before = before.pc_data
        after = after.pc_data
    print('Before and after shape after cloud retrieval:')
    print(before.shape)
    print(after.shape)

    before_z = before['z']
    print('Depth shape:')
    print(before_z.shape)
    after_z = after['z']

    time_benchmark = time.time()
    if bool_should_simulate_goal is True:
        pred_labels, final_contours, loaded_image = predictImageLabels(params_path, predict_num, im_path, ros_path,
            clf, dim_red_model, scaler)
    else:
        pred_labels, final_contours, loaded_image = predictImageLabels(params_path, predict_num, im_path, ros_path,
            clf, dim_red_model, scaler)
    print("Run time of Image pixel prediction: " + str(time.time() - time_benchmark) + " seconds.")
    time_benchmark = time.time()
    top_pool_id, top_pool_deepest_point_id, top_pool_volume = findOptimalDestination(before_z, after_z,
        image_size_rows, image_size_cols, pred_labels, final_contours, loaded_image, bool_display_final_contour)
    print("Run time of Optimal Dab Destination computation: " + str(time.time() - time_benchmark) + " seconds.")

    res = None
    if bool_should_simulate_goal is True:
        res = pixel_to_map_coordinates(top_pool_deepest_point_id, image_size_cols, image_size_rows)
    else:
        res = cloud_to_map_coordinates(top_pool_deepest_point_id, before)
    return res

def initialize_robot_arms(init_joint_state, blue_group, red_group, pub_gripper):
    #reset_start_state(red_group, red_group.get_active_joints())
    open_gripper(pub_gripper)
    rospy.sleep(1.0)
    _,__ = plan_and_move(init_joint_state, blue_group, None, None, 10, blue_group.get_active_joints(), 0)

    #Catastrophic failure bug, ignore initializing red for now
    '''red_group.go(init_joint_state, wait=True)
    red_group.stop()
    red_group.clear_pose_targets()'''

def open_gripper(publisher):
    move_gripper(1.0, publisher)

def close_gripper(publisher):
    move_gripper(0.0, publisher)

def move_gripper(pos, publisher):
    close_command = GripperCmd()
    close_command.emergency_release = False
    close_command.emergency_release_dir = 0
    close_command.stop = False
    close_command.position = pos
    close_command.speed = 0.0
    close_command.force = 0.0
    publisher.publish(close_command)


location_dab = [1.1, 0.3, -0.5]
joints_dab = [-1.1886,-1.7073,1.3124,-1.2104,1.5275,-0.4186]
joints_disposal = [-1.791,-1.8945,1.4523,-1.1339,1.5756,0.171]
joints_dab_pickup = [-1.1913,-1.7066,1.4022,-1.311,1.5275,-0.4185]
ros_path = "./src/surgical_dabber/src/"
plan_path = ros_path + 'Plans/'
params_path = ros_path+"params.json"
predict_num = 200
image_size_rows = 250
image_size_cols = 330
destination_x_offset = 0.5
bool_display_final_contour = True
im_path = ros_path + "Dataset/Processed/"
init_joint_state=[-0.20639741146209817,-0.15102010932503235,1.1829538210561723,0.5388390898820403,1.5708008621132574,1.3643624244936943]
bool_should_simulate_goal = True
classifier_name = ros_path+"blood_classifier.joblib"
pca_name = ros_path+"pca.joblib"
scaler_name = ros_path+"scaler.joblib"
dabbing_goal_test = []

with open(params_path) as json_data_file:  
    data = json.load(json_data_file)
    if 'location_dab' in data:
        location_dab = data['location_dab']
    if 'joints_disposal' in data:
        joints_disposal = data['joints_disposal']
    if 'joints_dab' in data:
        joints_dab = data['joints_dab']
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
    if 'destination_x_offset' in data:
        destination_x_offset = data['destination_x_offset']
    if 'init_joint_state' in data:
        init_joint_state = data['init_joint_state']
    if 'bool_should_simulate_goal' in data:
        bool_should_simulate_goal = data['bool_should_simulate_goal']
    if 'classifier_name' in data:
        classifier_name = ros_path+data['classifier_name']
    if 'pca_name' in data:
        pca_name = ros_path+data['pca_name']
    if 'scaler_name' in data:
        scaler_name = ros_path+data['scaler_name']
    if 'joints_dab_pickup' in data:
        joints_dab_pickup = data['joints_dab_pickup']
    if 'dabbing_goal_test' in data:
	dabbing_goal_test = data['dabbing_goal_test']
        
clf = load(classifier_name) 
dim_red_model = load(pca_name)
scaler = load(scaler_name)

pub_gripper = rospy.Publisher('/gripper/cmd', GripperCmd, queue_size=10)
moveit_commander.roscpp_initialize(sys.argv)
rospy.init_node('task1', anonymous=True)
scene = moveit_commander.PlanningSceneInterface()
robot = moveit_commander.RobotCommander()
tf_buffer = tf2_ros.Buffer()
tf_listener = tf2_ros.TransformListener(tf_buffer)

print("Robot group frames list:")
print(robot.get_group_names())
print("Robot state:")
print(robot.get_current_state())
group_name = robot.get_group_names()[0]
blue_group = moveit_commander.MoveGroupCommander(group_name)
red_group = moveit_commander.MoveGroupCommander('red_arm')
blue_group.set_planning_time(10.0)
red_group.set_planning_time(10.0)
blue_offset_x = -0.745


map_pub = rospy.Publisher('/map_blood', MarkerArray, queue_size = 100)

#insert_box(0+blue_offset_x, 0, -0.4, 4.0, 1.0, 0.01, scene, "ceiling", robot.get_planning_frame())
#insert_box(0+blue_offset_x, 0, 1.4, 4.0, 0.9, 0.01, scene, "floor", robot.get_planning_frame())
#insert_box(1.3+blue_offset_x, -0.35, -0.6, 0.2, 0.2, 1.0, scene, "path_obstacle", robot.get_planning_frame())
insert_box(0.7+blue_offset_x, 0.85, 1.2, 0.4, 0.25, 0.6, scene, "surgeon_body", robot.get_planning_frame())
insert_box(0.7+blue_offset_x, 0.85, 0.8, 0.2, 0.15, 0.2, scene, "surgeon_head", robot.get_planning_frame())
insert_box(0.45+blue_offset_x, 0.7, 1.1, 0.15, 0.4, 0.15, scene, "surgeon_left_arm", robot.get_planning_frame())
insert_box(0.95+blue_offset_x, 0.7, 1.1, 0.15, 0.4, 0.15, scene, "surgeon_right_arm", robot.get_planning_frame())
#insert_box(0, 0, 0.2, 0.025, 0.025, 0.2, scene, "dab", blue_group.get_end_effector_link())
insert_mesh(location_dab[0]+blue_offset_x, location_dab[1], location_dab[2], 0.25, 0.25, -0.25, scene, "dab_mesh",
            robot.get_planning_frame(), './src/surgical_dabber/src/Dataset/Processed/dab.stl')
rospy.sleep(1)

active_joints = blue_group.get_active_joints()
print('Active joints')
print(active_joints)

touch_links = robot.get_link_names(group=robot.get_group_names()[1])
#scene.attach_box(blue_group.get_end_effector_link(), "dab", touch_links=touch_links)
#blue_group.allow_looking(True)
#blue_group.allow_replanning(True)
#blue_group.set_planning_time(30.0)
#blue_group.set_num_planning_attempts(8)
print('Known constraints')
print(blue_group.get_known_constraints())

print("Planning frame name: " + str(blue_group.get_planning_frame()))
print("Endeffector link name: " + str(blue_group.get_end_effector_link()))
display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path', moveit_msgs.msg.DisplayTrajectory, queue_size=20)
#blue_group.pick("dab-mesh")
initialize_robot_arms(init_joint_state, blue_group, red_group, pub_gripper)
raw_input("Press Any Key to continue...")

prior_pcd = None
prior_depth = None
bridge = CvBridge()
if bool_should_simulate_goal is False:
    prior_pcd = rospy.wait_for_message('/camera/depth_registered/points', PointCloud2)
    trans_init = tf_buffer.lookup_transform(blue_group.get_planning_frame(), prior_pcd.header.frame_id,
                                            rospy.Time(0), rospy.Duration(1.0))
    rospy.sleep(1.0)
    prior_pcd = do_transform_cloud(prior_pcd, trans_init)
    prior_pcd = pypcd.PointCloud.from_msg(prior_pcd)
    prior_depth = rospy.wait_for_message('/camera/depth/image_rect_raw', Image)
    prior_depth = bridge.imgmsg_to_cv2(prior_depth, "32FC1")

    prior_depth = np.array(prior_depth, dtype = np.dtype('f8'))
    prior_depth = cv2.normalize(prior_depth, prior_depth, 0, 1, cv2.NORM_MINMAX)

    cv2.imshow('Prior depth map', prior_depth)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

loop_counter = 0
time_loop = 0
while not rospy.is_shutdown():
    time_loop = time.time()
    if loop_counter == 0:
        _,__ = plan_and_move(joints_dab, blue_group, plan_path, 'idle_to_dab', 20, active_joints, 0)
    else:
        _,__ = plan_and_move(joints_dab, blue_group, plan_path, 'disposal_to_dab', 10, active_joints, 0)

    scene.attach_mesh(blue_group.get_end_effector_link(), "dab_mesh", touch_links=touch_links)
    close_gripper(pub_gripper)
    #Wait for the gripper to close
    rospy.sleep(3.0)
    raw_input("Press Any Key to continue...")

    _,__ = plan_and_move(joints_dab_pickup, blue_group, plan_path, 'dab_to_dab_pickup', 10, active_joints, 0)
    _,__ = plan_and_move(init_joint_state, blue_group, plan_path, 'dab_pickup_to_idle', 10, active_joints, 0)
    rospy.sleep(1.0)
    
    dabbing_goal = None
    posterior_pcd = None
    cleaning_goal = None
    cleaning_plan = None
    if bool_should_simulate_goal is False:
        posterior_pcd = rospy.wait_for_message('/camera/depth_registered/points', PointCloud2)
        trans_out = tf_buffer.lookup_transform(blue_group.get_planning_frame(), posterior_pcd.header.frame_id,
                                               rospy.Time(0), rospy.Duration(1.0))
	post_depth = rospy.wait_for_message('/camera/depth/image_rect_raw', Image)
        rospy.sleep(1.0)
        posterior_pcd = do_transform_cloud(posterior_pcd, trans_out)
        posterior_pcd = pypcd.PointCloud.from_msg(posterior_pcd)
        dab_img = rospy.wait_for_message('/camera/color/image_rect_color', Image)
        dab_img = bridge.imgmsg_to_cv2(dab_img, 'passthrough')
        post_depth = bridge.imgmsg_to_cv2(post_depth, '32FC1')

        post_depth = np.array(post_depth, dtype = np.dtype('f8'))
        post_depth = cv2.normalize(post_depth, post_depth, 0, 1, cv2.NORM_MINMAX)
        cv2.imshow('Posterior depth map', post_depth)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print('Live camera image shape:')
        print(dab_img.shape)
        image_size_rows = len(dab_img)
        image_size_cols = len(dab_img[0])
        dabbing_goal = extract_optimal_goal(prior_pcd, posterior_pcd, dab_img, image_size_cols, 
            image_size_rows, im_path, params_path, ros_path, bool_display_final_contour, bool_should_simulate_goal,
            clf, dim_red_model, scaler)
        print('Final dabbing goal:')
        dabbing_goal[2] = dabbing_goal[2]-0.31
        print(dabbing_goal)
	raw_input("Press Any Key to continue...")
	cleaning_goal, cleaning_plan = plan_and_move(dabbing_goal, blue_group, plan_path, None, 20, 		    active_joints, 0)
    else:
        input_pcd = int(raw_input("Point cloud index (1-3 int):"))
        input_predict_num = int(raw_input("Predicted image number (1-200 int):"))
        destination_marker = create_mesh_marker(blue_offset_x+destination_x_offset, 0.0, 1.39, 0.33, 0.33, -11.0,
            robot.get_planning_frame(), 'package://surgical_dabber/src/Dataset/Processed/body_'+str(input_pcd)+'.dae')
        map_pub.publish(destination_marker)
        rospy.sleep(0.01)

        before_path = im_path+'pc_mock/before_'+str(input_pcd)+'.PCD'
        after_path = im_path+'pc_mock/after_'+str(input_pcd)+'.PCD'

        dabbing_goal = extract_optimal_goal(before_path, after_path, input_predict_num, image_size_cols, 
            image_size_rows, im_path, params_path, ros_path, bool_display_final_contour, bool_should_simulate_goal,
            clf, dim_red_model, scaler)
	cleaning_goal, cleaning_plan = plan_and_move(dabbing_goal, blue_group, plan_path, None, 20, 		    active_joints, blue_offset_x+destination_x_offset)
	

    
    cleaning_plan[1].joint_trajectory.points.reverse()
    
    has_reached = check_if_goal_reached(cleaning_goal, blue_group, 0.01)

    rospy.sleep(2.0)
    raw_input("Press Any Key to continue...")

    #This has no backup trajectory, reverse plan is used, however, it is not functional at the moment
    #_,__ = plan_and_move(init_joint_state, blue_group, plan_path, None, 10, active_joints, blue_offset_x, cleaning_plan[1])
    _,__ = plan_and_move(init_joint_state, blue_group, plan_path, None, 10, active_joints, 0)

    _,__ = plan_and_move(joints_disposal, blue_group, plan_path, 'idle_to_disposal', 10, active_joints, 0)
    
    scene.remove_attached_object(blue_group.get_end_effector_link(), "dab_mesh")
    scene.remove_world_object("dab_mesh")
    insert_mesh(location_dab[0]+blue_offset_x, location_dab[1], location_dab[2], 0.25, 0.25, -0.25, scene, "dab_mesh",
                robot.get_planning_frame(), './src/surgical_dabber/src/Dataset/Processed/dab.stl')
    rospy.sleep(0.1)

    loop_counter += 1

    print('Whole loop time taken: ' + str(time_loop-time.time()))

#dump(plan_4, "idle_to_body")

'''pose_test_joints = blue_group.get_current_joint_values()
pose_test_joints[5] += 0.75
print(pose_test_joints)

blue_group.go(pose_test_joints, wait=True)
blue_group.stop()
blue_group.clear_pose_targets()

rospy.sleep(1)'''
    
