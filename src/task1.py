#!/usr/bin/env python

import sys
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import visualization_msgs
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from moveit_commander.conversions import pose_to_list
from math import pi

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
    
def plan_to_goal(x, y, z, cur_group):
    pose_goal = geometry_msgs.msg.Pose()
    pose_goal.orientation.x = -1.0
    #pose_goal.orientation.w = 1.0
    pose_goal.position.x = x
    pose_goal.position.y = y
    pose_goal.position.z = z
    cur_group.set_pose_target(pose_goal)
    plan = cur_group.plan()
    return plan, pose_goal
    
def check_if_goal_reached(pose_goal, cur_group, tolerance):
    pose_goal_arr = pose_to_list(pose_goal)
    cur_pose = pose_to_list(cur_group.get_current_pose().pose)

    bool_reached_goal = True
    for i in range(len(cur_pose)):
        if (abs(cur_pose[i] - pose_goal_arr[i]) > tolerance):
            bool_reached_goal = False

    print('Did robot reach goal within tolerance: ' + str(bool_reached_goal))
    return bool_reached_goal
    
location_dab = [0.0, 0.0, -0.3]
location_idle = [1.1, 0.3, -0.5]
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
group_name = robot.get_group_names()[1]
blue_group = moveit_commander.MoveGroupCommander(group_name)
blue_group.set_planning_time(10.0)


map_pub = rospy.Publisher('/map_blood', MarkerArray, queue_size = 100)
destination_marker = create_mesh_marker(0.0, 0.0, -0.75, 1.0, 1.0, 1.0, robot.get_planning_frame(), 'package://scripts/blood_dabbing.dae')
map_pub.publish(destination_marker)
rospy.sleep(1)



insert_box(0, 0, 0.6, 0.75, 0.75, 0.1, scene, "ceiling", robot.get_planning_frame())
insert_box(0, 0, -0.76, 2.0, 2.0, 0.01, scene, "floor", robot.get_planning_frame())
#insert_box(1.3, -0.35, -0.5, 0.2, 0.2, 1.0, scene, "path_obstacle", robot.get_planning_frame())
insert_box(-1.15, 0.5, -0.5, 0.25, 0.4, 0.6, scene, "surgeon_body", robot.get_planning_frame())
insert_box(-1.15, 0.5, -0.1, 0.15, 0.2, 0.2, scene, "surgeon_head", robot.get_planning_frame())
insert_box(-1.0, 0.25, -0.4, 0.4, 0.15, 0.15, scene, "surgeon_left_arm", robot.get_planning_frame())
insert_box(-1.0, 0.75, -0.4, 0.4, 0.15, 0.15, scene, "surgeon_right_arm", robot.get_planning_frame())
#insert_box(0, 0, 0.2, 0.025, 0.025, 0.2, scene, "dab", blue_group.get_end_effector_link())
insert_box(0, 0, 0.2, 0.025, 0.025, 0.2, scene, "dab", blue_group.get_end_effector_link())
#insert_mesh(dab_goal.position.x, dab_goal.position.y, dab_goal.position.z, 1.0, 1.0, 1.0, scene, "dab_mesh", robot.get_planning_frame(), 'dab.stl')
rospy.sleep(1)



touch_links = robot.get_link_names(group=robot.get_group_names()[0])
scene.attach_box(blue_group.get_end_effector_link(), "dab", touch_links=touch_links)

print("Planning frame name: " + str(blue_group.get_planning_frame()))
print("Endeffector link name: " + str(blue_group.get_end_effector_link()))
display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path', moveit_msgs.msg.DisplayTrajectory, queue_size=20)

plan_1, plan_1_pose = plan_to_goal(location_dab[0], location_dab[1], location_dab[2], blue_group)
pose_goal_joints = blue_group.get_current_joint_values()

blue_group.go(wait=True)
blue_group.stop()
blue_group.clear_pose_targets()

rospy.sleep(1)

plan_1, plan_1_pose = plan_to_goal(location_idle[0], location_idle[1], location_idle[2], blue_group)
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
blue_group.clear_pose_targets()


pose_test_joints = blue_group.get_current_joint_values()
print(pose_test_joints)
has_reached = check_if_goal_reached(plan_1_pose, blue_group, 0.01)

while not rospy.is_shutdown():
    map_pub.publish(destination_marker)
    rospy.sleep(0.01)
