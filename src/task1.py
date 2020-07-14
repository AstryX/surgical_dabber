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

moveit_commander.roscpp_initialize(sys.argv)
rospy.init_node('task1', anonymous=True)
scene = moveit_commander.PlanningSceneInterface()
robot = moveit_commander.RobotCommander()
group_name = "ur10"
blue_group = moveit_commander.MoveGroupCommander(group_name)
blue_group.set_planning_time(30.0)

marker_array = MarkerArray()

map_pub = rospy.Publisher('/map_blood', MarkerArray, queue_size = 100)

marker = Marker()
#marker.type = marker.SPHERE
marker.type = marker.MESH_RESOURCE
marker.mesh_resource = 'package://scripts/blood_dabbing.dae'
marker.header.frame_id = robot.get_planning_frame()
marker.action = 0
marker.header.stamp = rospy.get_rostime()
marker.id = 0
marker.pose.orientation.w = 1.0
marker.scale.x = 1.0
marker.scale.y = 1.0
marker.scale.z = 1.0
#marker.color.r = 0.0
#marker.color.g = 1.0
#marker.color.b = 1.0
#marker.color.a = 1.0
marker.mesh_use_embedded_materials = True
marker.pose.position.x = 0
marker.pose.position.y = 0
marker.pose.position.z = -0.75

marker_array.markers.append(marker)

rospy.sleep(2)

map_pub.publish(marker_array)
rospy.sleep(1)

print("Robot group frames list:")
print(robot.get_group_names())
print("Robot state:")
print(robot.get_current_state())


rospy.sleep(2)

box_pose = geometry_msgs.msg.PoseStamped()
box_pose.header.frame_id = robot.get_planning_frame()
box_pose.pose.position.x = 0.0
box_pose.pose.position.y = 0.0
box_pose.pose.position.z = 0.5
box_pose.pose.orientation.w = 1.0
box_name = "obstacle"
scene.add_box(box_name, box_pose, size=(0.5, 0.5, 0.2))

rospy.sleep(2)

box_pose2 = geometry_msgs.msg.PoseStamped()
box_pose2.header.frame_id = blue_group.get_end_effector_link()
box_pose2.pose.orientation.w = 1.0
box_pose2.pose.position.z = 0.175
box_name2 = "dab"
scene.add_box(box_name2, box_pose2, size=(0.025, 0.025, 0.175))

rospy.sleep(2)

touch_links = robot.get_link_names(group='end_effector')
scene.attach_box(blue_group.get_end_effector_link(), "dab", touch_links=touch_links)

print("Planning frame name: " + str(blue_group.get_planning_frame()))
print("Endeffector link name: " + str(blue_group.get_end_effector_link()))
display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path', moveit_msgs.msg.DisplayTrajectory, queue_size=20)

pose_goal = geometry_msgs.msg.Pose()
pose_goal.orientation.x = -1.0
#pose_goal.orientation.w = 1.0
pose_goal.position.x = 0.7
pose_goal.position.y = -0.5
pose_goal.position.z = -0.5
blue_group.set_pose_target(pose_goal)


blue_plan = blue_group.go(wait=True)

blue_group.stop()
blue_group.clear_pose_targets()

pose_goal_arr = pose_to_list(pose_goal)
cur_pose = pose_to_list(blue_group.get_current_pose().pose)

bool_reached_goal = True
tolerance = 0.01
for i in range(len(cur_pose)):
    if (abs(cur_pose[i] - pose_goal_arr[i]) > tolerance):
        bool_reached_goal = False

print('Did robot reach goal within tolerance: ' + str(bool_reached_goal))

while not rospy.is_shutdown():
    map_pub.publish(marker_array)
    rospy.sleep(0.01)
