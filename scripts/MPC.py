#!/usr/bin/env python 

import rospy
from Include.classes_base import *
from math import sqrt

start = False
L = 1.2
if rospy.has_param("L_formation"):
    L= rospy.get_param("L_formation")

target_points ={"rosie0": [L/sqrt(3),0],\
                "rosie1" : [-L/(2*sqrt(3)),L/2],\
                "rosie2" : [-L/(2*sqrt(3)),-L/2] } 

### INIZIALIZATION
# rosie0 = Object("rosie0","leader")
# rosie1 = Object("rosie1","follower")
# rosie2 = Object("rosie2","follower")
# rosies=[rosie0,rosie1,rosie2]

ns = rospy.get_namespace()
name = rospy.get_param(ns+"/rosie_name")
# for rosie in rosies:
#     if name==rosie.name:
#         rosieX = rosie
#         rosieX.pos_wrt_target = target_points[name]
if name == "rosie0":
    obj_type = "leader"
else:
     obj_type = "follower"
print(name)
rosieX = Object(name,obj_type,target_points[name])
Big_box = Object("Big_box","target")

print ("inizialiazation")

print("object {} is ready".format(rosieX.name))


def move_all_pub():

    while not rospy.is_shutdown():
        if rosieX.start_target and rosieX.pos_bool and Big_box.pos_bool:
            rosieX.MPC_controller(Big_box,L)#,rosies)

        else:
            rospy.sleep(1)
            print(rosieX.name)
            print("big_box pos:",Big_box.pos_bool,"odom: ",rosieX.pos_bool)
            print("move base :", rosieX.start_target)

if __name__ == '__main__':
    rospy.init_node('MPC', anonymous=True)
    try:
        # for rosie in rosies:
        #     rosie.sub()
        rosieX.sub_formation_init()
        rosieX.sub()
        rosieX.sub_target_pos()
        rosieX.sub_formation_pose()
        Big_box.sub()
        move_all_pub()
        rospy.spin()   

    except rospy.ROSInterruptException:
        pass

