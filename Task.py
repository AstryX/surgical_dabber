#!/usr/bin/env python

import sys
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg


moveit_commander.roscpp_initialize(sys.argv)
rospy.init_node('task1', anonymous=True)
robot = moveit_commander.RobotCommander()

group_name = "ur10"
blue_group = moveit_commander.MoveGroupCommander(group_name)
display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path', moveit_msgs.msg.DisplayTrajectory, queue_size=20)

pose_goal = geometry_msgs.msg.Pose()
pose_goal.orientation.x = -1.0
#pose_goal.orientation.w = 1.0
pose_goal.position.x = 0
pose_goal.position.y = 0.35
pose_goal.position.z = -1.15
blue_group.set_pose_target(pose_goal)


blue_plan = blue_group.go(wait=True)


#joint_vals=moveit_commander.move_group.MoveGroupCommander.get_current_joint_values(red_group)
#print(joint_vals)
#joint_vals[1]=-1.4378
#red_group.set_joint_value_target(joint_vals)
#red_plan=red_group.go(wait=True)

#pose_goal.position.x = 0.1
#pose_goal.position.y = 0
#pose_goal.position.z = 0.8
#red_group.set_pose_target(pose_goal)
#red_plan = red_group.go(wait=True)
#red_group.stop()
#blue_group.stop()
#red_group.clear_pose_targets()
#blue_group.clear_pose_targets()
#red_group.execute(red_plan, wait=True)
#blue_group.execute(blue_plan, wait=True)
#joints = rospy.Publisher('/move_group/joints', std_msgs.msg, queue_size=20)
