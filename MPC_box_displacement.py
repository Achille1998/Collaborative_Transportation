#!/usr/bin/env python
# Input --> target pos
# Output --> force for the formation

from casadi import *
import rospy
from Include.classes_base import *
from Include.classes_box_for_all import *
from Include.classes_formation import Formation
from Include.classes_arm import Arm
#dimension of the triangle
L = 1
#create all objects
rosie0 = Object("rosie0","leader") 
rosie1 = Object("rosie1","follower") 
rosie2 = Object("rosie2","follower") 
target_object = Object("Big_box","target")
G = np.array([[0.16,0,0],[-0.05,0.24,0],[-0.05,-0.21,0]]).T
#formation = Formation([],target_object)
formation = Formation([rosie0,rosie1,rosie2],target_object)

Arm0 = Arm("Arm0",pub_bool=False)
Arm1 = Arm("Arm1",pub_bool=False)
Arm2 = Arm("Arm2",pub_bool=False)


Big_box = Box( formation, [Arm0,Arm1,Arm2], G)


def talker():
    rate = rospy.Rate(100)
    while not rospy.is_shutdown():
        if Big_box.start_odom and Big_box.start_disp:
            Big_box.MPC_controller()
        else:
            
            # print("MPC displacement ----> vel")
            # print("odometry :",Big_box.start_odom)
            # print("displacement :",Big_box.start_disp)
            formation.compute_pos()
            rate.sleep()
            
            
if __name__ == '__main__':
    try:
        rospy.init_node('MPC_box_displacement', anonymous=True)
        
        Big_box.sub()
        
        talker()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


