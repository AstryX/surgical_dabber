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
from joblib import dump,load
import pyassimp
from math import pi
import json

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
        pose_goal.orientation.x = -1.0
        #pose_goal.orientation.w = 1.0
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
    
location_idle = [0.0, 0.0, -0.5]
location_dab = [1.1, 0.3, -0.5]
location_disposal = [0.3, -1.1, -0.5]
    
ros_path = "./src/surgical_dabber/src/"
params_path = ros_path+"params.json"

with open(params_path) as json_data_file:  
    data = json.load(json_data_file)
    if 'location_dab' in data:
        location_dab = data['location_dab']
    if 'location_idle' in data:
        location_idle = data['location_idle']
    if 'location_disposal' in data:
        location_disposal = data['location_disposal']
    
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
blue_group.set_planning_time(10.0)
blue_offset_x = -0.745


map_pub = rospy.Publisher('/map_blood', MarkerArray, queue_size = 100)
destination_marker = create_mesh_marker(0.0, 0.0, -1.25, 1.0, 1.0, 1.0, robot.get_planning_frame(), 'package://scripts/blood_dabbing.dae')
map_pub.publish(destination_marker)
rospy.sleep(1)



#insert_box(0, 0, 0.6, 0.75, 0.75, 0.1, scene, "ceiling", robot.get_planning_frame())
insert_box(0+blue_offset_x, 0, -1.26, 2.0, 2.0, 0.01, scene, "floor", robot.get_planning_frame())
#insert_box(1.3, -0.35, -0.9, 0.2, 0.2, 1.0, scene, "path_obstacle", robot.get_planning_frame())
insert_box(-1.15+blue_offset_x, 0.5, -1.0, 0.25, 0.4, 0.6, scene, "surgeon_body", robot.get_planning_frame())
insert_box(-1.15+blue_offset_x, 0.5, -0.6, 0.15, 0.2, 0.2, scene, "surgeon_head", robot.get_planning_frame())
insert_box(-1.0+blue_offset_x, 0.25, -0.9, 0.4, 0.15, 0.15, scene, "surgeon_left_arm", robot.get_planning_frame())
insert_box(-1.0+blue_offset_x, 0.75, -0.9, 0.4, 0.15, 0.15, scene, "surgeon_right_arm", robot.get_planning_frame())
#insert_box(0, 0, 0.2, 0.025, 0.025, 0.2, scene, "dab", blue_group.get_end_effector_link())
#insert_box(0, 0, 0.2, 0.025, 0.025, 0.2, scene, "dab", blue_group.get_end_effector_link())
insert_mesh(location_dab[0]+blue_offset_x, location_dab[1], location_dab[2], 1.0, 1.0, 1.0, scene, "dab_mesh", robot.get_planning_frame(), './src/surgical_dabber/src/Dataset/Processed/dab.stl')
rospy.sleep(1)



touch_links = robot.get_link_names(group=robot.get_group_names()[1])
#scene.attach_box(blue_group.get_end_effector_link(), "dab", touch_links=touch_links)
blue_group.allow_looking(True)
blue_group.allow_replanning(True)
blue_group.set_planning_time(30.0)
blue_group.set_num_planning_attempts(8)
print('Known constraints')
print(blue_group.get_known_constraints())
pose_start_joints = blue_group.get_current_joint_values()
print('Start joints:')
print(pose_start_joints)
#blue_group.set_start_state_to_current_state()

print("Planning frame name: " + str(blue_group.get_planning_frame()))
print("Endeffector link name: " + str(blue_group.get_end_effector_link()))
display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path', moveit_msgs.msg.DisplayTrajectory, queue_size=20)

#blue_group.pick("dab-mesh")

plan_1, plan_1_pose = plan_to_goal(location_idle[0]+blue_offset_x, location_idle[1], location_idle[2], blue_group, 10, ros_path+'start_to_idle')
print(plan_1)
blue_group.execute(plan_1[1], wait=True)
blue_group.stop()
blue_group.clear_pose_targets()


rospy.sleep(1)

#blue_group.set_start_state_to_current_state()
pose_goal_joints = blue_group.get_current_joint_values()
print('Cur joints')
print(pose_goal_joints)
active_joints = blue_group.get_active_joints()
print('Active joints')
print(active_joints)

joint_state = JointState()
joint_state.header = Header()
joint_state.header.stamp = rospy.Time.now()
joint_state.name = active_joints
joint_state.position = pose_goal_joints
moveit_robot_state = RobotState()
moveit_robot_state.joint_state = joint_state
blue_group.set_start_state(moveit_robot_state)

plan_2, plan_2_pose = plan_to_goal(location_dab[0]+blue_offset_x, location_dab[1], location_dab[2]+0.4, blue_group, 10, ros_path+'idle_to_dab')
blue_group.execute(plan_2[1], wait=True)
blue_group.stop()
blue_group.clear_pose_targets()

scene.attach_mesh(blue_group.get_end_effector_link(), "dab_mesh", touch_links=touch_links)

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

plan_3, plan_3_pose = plan_to_goal(location_idle[0]+blue_offset_x, location_idle[1], location_idle[2], blue_group, 10, ros_path+'dab_to_idle')
blue_group.execute(plan_3[1], wait=True)
blue_group.stop()
blue_group.clear_pose_targets()

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

plan_4, plan_4_pose = plan_to_goal(0.7+blue_offset_x, -0.5, -0.5, blue_group, 10, ros_path+'idle_to_body')
blue_group.execute(plan_4[1], wait=True)
blue_group.stop()
blue_group.clear_pose_targets()

#dump(plan_4, "idle_to_body")

'''pose_test_joints = blue_group.get_current_joint_values()
pose_test_joints[5] += 0.75
print(pose_test_joints)

blue_group.go(pose_test_joints, wait=True)
blue_group.stop()
blue_group.clear_pose_targets()

rospy.sleep(1)'''


'''plan_1, plan_1_pose = plan_to_goal(location_idle[0], location_idle[1], location_idle[2], blue_group)
pose_goal_joints = blue_group.get_current_joint_values()

blue_group.go(wait=True)
blue_group.stop()
blue_group.clear_pose_targets()

rospy.sleep(1)

plan_1, plan_1_pose = plan_to_goal(0.7, -0.5, -0.4, blue_group)
pose_goal_joints = blue_group.get_current_joint_values()

blue_group.go(wait=True)
blue_group.stop()
blue_group.clear_pose_targets()

rospy.sleep(1)

plan_1, plan_1_pose = plan_to_goal(location_disposal[0], location_disposal[1], location_disposal[2], blue_group)
pose_goal_joints = blue_group.get_current_joint_values()

blue_group.go(wait=True)
blue_group.stop()
blue_group.clear_pose_targets()'''

has_reached = check_if_goal_reached(plan_4_pose, blue_group, 0.01)

while not rospy.is_shutdown():
    map_pub.publish(destination_marker)
    rospy.sleep(0.01)
