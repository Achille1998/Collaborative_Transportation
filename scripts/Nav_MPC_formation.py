#!/usr/bin/env python
from math import cos, sin,pi
import numpy as np
import rospy
from Include.classes_base import *
from Include.classes_formation_with_obst_avoidance import *
from square_move.msg import formation_msg

Q_x = 1100
Q_y = 1100
Q_theta = 0
# Input matrix weights
R1 = 700
R2 = 700
R3 = 300
weights = [Q_x,Q_y,Q_theta,R1,R2,R3]
### INIZIALIZATION

rosie0 = Object("rosie0","leader")
rosie1 = Object("rosie1","follower") 
rosie2 = Object("rosie2","follower") 
Big_box = Object("Big_box","target")
human = Object("human","obstacle") 
ball = Object("ball","obstacle")
obstacles =[]
for obj in all_object:
    if obj.type == "obstacle":
        obstacles.append(obj)


formation = Formation([rosie0,rosie1,rosie2],Big_box,weights,
                        max_lin_disp=0.06,
                        max_ang_disp=pi/32,
                        pub_bool=False) #let MPC_box pub formation_pose


print ("MPC NAV formation inizialiazation")
print("formation is ready:")
for i,obj in enumerate(formation.obj_members):
    print("member {}: {}".format(i,obj.name))

def Move_formation():
    rate = rospy.Rate(10)

    while not rospy.is_shutdown():
        
        if formation.start: #target position received
            formation.MPC_controller(obstacles)
        else :
            print(" ------------Waiting for - target position --------")
            print("send target position at /box/target_pos, target_pos")
        
            rospy.sleep(1)


        
if __name__ == '__main__':
    rospy.init_node('Nav_MPC_formation', anonymous=True)
    try:
        #subscribers
        for obj in all_object:
            obj.sub() #update pos

        formation.sub() # get target pos
        
        Move_formation()
        rospy.spin()   

    except rospy.ROSInterruptException:
        pass

