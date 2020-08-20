# surgical_dabber
The following is created by University of Edinburgh MSc Artificial Intelligence student Robertas Dereskevicius for his
Robotic Surgical Dabbing thesis. The contents of this submission are incomplete, as the full project would take up over 20 gigabytes.
It contains all of the necessary source code and data set/masks in source/saifer-surgery/src/surgical_dabber folder. The remaining
folders are configurations of the saifer-surgery repository which has been configured to work with a surgical dabber.\
Student dabber source git: https://github.com/AstryX/surgical_dabber \
Saifer-surgery git: https://github.com/ipab-rad/saifer-surgery \
There are large libraries such as moveit, stomp, etc missing. \
Contact astryxroberto@gmail.com if you require support and want to enquire about the Robotic Surgical Dabber.\
The complete system is built using: catkin build in saifer-surgery folder.

The simulation stack is run:\
source devel/setup.bash\
roslaunch dual_moveit demo.launch\
2nd terminal: source again\
rosrun surgical_dabber task1.py

Live dabbing is run:\
source devel/setup.bash\
roslaunch saifer_launch dual_arm.launch\
2nd terminal: source again\
roslaunch realsense2_camera rs_rgbd.launch\
3rd terminal: source again\
rosrun surgical_dabber task1.py
