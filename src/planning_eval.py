#!/usr/bin/env python

import sys
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from moveit_msgs.msg import RobotState
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
import visualization_msgs
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from moveit_commander.conversions import pose_to_list
import json
import math
import time

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

def plan_to_goal(x, y, z, cur_group, num_tries, backup_plan):
    full_plan = None
    total_tries = 0
    for i in range(num_tries):
        total_tries += 1
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

    return full_plan, pose_goal, total_tries

def plan_to_goal_joints(joints, cur_group, num_tries, backup_plan):
    full_plan = None
    total_tries = 0
    for i in range(num_tries):
        total_tries += 1
        cur_group.set_joint_value_target(joints)
        plan = cur_group.plan()

        if plan[0] == True:
            print('Path found after ' + str(i+1) + ' out of ' + str(num_tries) + ' attempts!')
            full_plan = plan
            break

    if ((full_plan is None) and (backup_plan is not None)):
        print('Path not found after ' + str(num_tries) + ' tries! Reverting to the backup choice...')
        full_plan = load(backup_plan)

    return full_plan, None, total_tries
    
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
    #rospy.sleep(1)
    #reset_start_state(blue_group, active_joints)
    combined_path = None
    plan_pose = None
    plan = None
    plan_time = time.time()
    fail_times = 0
    total_tries = 0
    if precomputed_safe_plan is None:
        if backup_plan is not None:
            combined_path = ros_path+backup_plan
        if len(location_array) == 3:
            plan, plan_pose, total_tries = plan_to_goal(location_array[0]+x_offset, location_array[1], location_array[2], blue_group, num_tries, combined_path)
        else:
            plan, plan_pose, total_tries = plan_to_goal_joints(location_array, blue_group, num_tries, combined_path)
        #Saving plans code:
        #if (plan is not None) and (backup_plan is not None):
        #    dump(plan, backup_plan)
        plan_time = time.time() - plan_time
        if plan is not None:
            #blue_group.execute(plan[1], wait=True)
            asd = 0
        else:
            fail_times = 1
            plan_time = 0
            total_tries = 0
    else:
        print('Executing precomputed safe plan!')
        blue_group.execute(precomputed_safe_plan, wait=True)
    blue_group.stop()
    blue_group.clear_pose_targets()

    return plan_pose, plan, plan_time, fail_times, total_tries

def initialize_robot_arms(init_joint_state, blue_group, red_group):
    reset_start_state(blue_group, blue_group.get_active_joints())
    reset_start_state(red_group, red_group.get_active_joints())

    blue_group.go(init_joint_state, wait=True)
    blue_group.stop()
    blue_group.clear_pose_targets()

moveit_commander.roscpp_initialize(sys.argv)
rospy.init_node('task1', anonymous=True)
scene = moveit_commander.PlanningSceneInterface()
robot = moveit_commander.RobotCommander()

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
benchmark_type = 1

map_pub = rospy.Publisher('/map_blood', MarkerArray, queue_size = 100)

active_joints = blue_group.get_active_joints()
print('Active joints')
print(active_joints)

touch_links = robot.get_link_names(group=robot.get_group_names()[1])
print('Known constraints')
print(blue_group.get_known_constraints())

print("Planning frame name: " + str(blue_group.get_planning_frame()))
print("Endeffector link name: " + str(blue_group.get_end_effector_link()))
display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path', moveit_msgs.msg.DisplayTrajectory, queue_size=20)

joints_dab=[-0.8503,-2.1689,1.9641,-1.3855,1.5248,-0.7529]
start_joint_values = blue_group.get_current_joint_values()

if benchmark_type == 1:
    placeholder = 0
    #insert_box(1.3+blue_offset_x, -0.35, -0.6, 0.2, 0.2, 1.0, scene, "path_obstacle", robot.get_planning_frame())
elif benchmark_type == 3:
    #insert_box(1.3+blue_offset_x, -0.35, -0.6, 0.2, 0.2, 1.0, scene, "path_obstacle", robot.get_planning_frame())
    insert_box(0.37+blue_offset_x,-0.35,1.2, 0.5, 0.03, 0.225, scene, "obs", robot.get_planning_frame())
    rospy.sleep(1)
    initialize_robot_arms(joints_dab, blue_group, red_group)
    start_joint_values = blue_group.get_current_joint_values()
    joints_dab=[-1.1886,-1.7073,1.3124,-1.2104,1.5275,-0.4186]
else:
    #insert_box(1.3+blue_offset_x, -0.35, -0.6, 0.2, 0.2, 1.0, scene, "path_obstacle", robot.get_planning_frame())
    rospy.sleep(1)
    initialize_robot_arms(joints_dab, blue_group, red_group)
    start_joint_values = blue_group.get_current_joint_values()
    joints_dab=[blue_offset_x+0.6,0.2,1.2]
    #insert_box(joints_dab[0], joints_dab[1], joints_dab[2], 0.1, 0.1, 0.1, scene, "mark", robot.get_planning_frame())
rospy.sleep(1)



num_tests = 50
total_fail = 0
total_tries = 0
total_time = 0
total_distance = 0
for i in range(num_tests):
    joint_state = JointState()
    joint_state.header = Header()
    joint_state.header.stamp = rospy.Time.now()
    joint_state.name = active_joints
    joint_state.position = start_joint_values
    moveit_robot_state = RobotState()
    moveit_robot_state.joint_state = joint_state
    blue_group.set_start_state(moveit_robot_state)
    rospy.sleep(0.1)

    plan_pose, plan, plan_time, fail_times, tries_plan = plan_and_move(joints_dab, blue_group, None, None, 10, active_joints, 0)
    total_delta = [0.0,0.0,0.0,0.0,0.0,0.0]
    if fail_times == 0:
        for j in range(len(plan[1].joint_trajectory.points)):
            if j == 0:
                continue
            else:
                total_delta[0] += abs(plan[1].joint_trajectory.points[j].positions[0] - plan[1].joint_trajectory.points[j-1].positions[0])
                total_delta[1] += abs(plan[1].joint_trajectory.points[j].positions[1] - plan[1].joint_trajectory.points[j-1].positions[1])
                total_delta[2] += abs(plan[1].joint_trajectory.points[j].positions[2] - plan[1].joint_trajectory.points[j-1].positions[2])
                total_delta[3] += abs(plan[1].joint_trajectory.points[j].positions[3] - plan[1].joint_trajectory.points[j-1].positions[3])
                total_delta[4] += abs(plan[1].joint_trajectory.points[j].positions[4] - plan[1].joint_trajectory.points[j-1].positions[4])
                total_delta[5] += abs(plan[1].joint_trajectory.points[j].positions[5] - plan[1].joint_trajectory.points[j-1].positions[5])

    total_distance += sum(total_delta)

    total_time += plan_time
    total_fail += fail_times
    total_tries += tries_plan
    #cleaning_goal, cleaning_plan = plan_and_move(dabbing_goal, blue_group, plan_path, None, 20, active_joints, blue_offset_x+destination_x_offset)
    #has_reached = check_if_goal_reached(cleaning_goal, blue_group, 0.01)
    rospy.sleep(0.1)

print('Planning experiment results!')
print('Average planning time iteration:')
print(float(total_time)/(num_tests-fail_times))
print('Average number of planning tries needed:')
print(float(total_tries)/(num_tests-fail_times))
print('Number of complete failures:')
print(total_fail)
print('Total experiment distance:')
print(float(total_distance)/(num_tests-fail_times))

'''pose_test_joints = blue_group.get_current_joint_values()
pose_test_joints[5] += 0.75
print(pose_test_joints)

blue_group.go(pose_test_joints, wait=True)
blue_group.stop()
blue_group.clear_pose_targets()

rospy.sleep(1)'''
    
