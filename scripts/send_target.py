#!/usr/bin/env python

import rospy
import numpy as np
from square_move.msg import target_pos,formation_msg
from math import pi
start = True
def talker():
   
    pub = rospy.Publisher("/box/target_pos", target_pos,queue_size=1)
    
    rate = rospy.Rate(20)
    T0 = rospy.Time.now().to_sec()
    w = 6.28/180 #5 minuts
    r = 0.5 #* np.sqrt(2)
    T =20
    i=0
    theta = pi/4
    while not rospy.is_shutdown():
        if start:
            t = rospy.Time.now().to_sec() - T0
            # if t > T:
            #     T0 = rospy.Time.now().to_sec()
            #     theta += pi/2
            theta = (t-5) * w
            target_position = target_pos()
            target_position.pos = np.array([r*np.cos(theta),r*np.sin(theta),0.1,1.57+theta]) # x,y,z,theta(z)
    
    # r = 0.5/ np.cos(pi/8)
    # T =30
    # i=0
    # theta = pi/8
    # while not rospy.is_shutdown():
    #     if start:
    #         t = rospy.Time.now().to_sec() - T0
    #         if t > T:
    #             T0 = rospy.Time.now().to_sec()
    #             theta += pi/4
            
    #         target_position = target_pos()
    #         target_position.pos = np.array([r*np.cos(theta),r*np.sin(theta),0.1,1.57+theta - pi/8]) # x,y,z,theta(z)
    
            print(t)
            pub.publish(target_position)
            rospy.loginfo(target_position)
            rate.sleep()

def callback(msg):
    global start
    if msg.init == "gripper_closed":
        start = True

        
if __name__ == '__main__':
    rospy.init_node('send_target', anonymous=True)
    try:
        rospy.Subscriber("/Formation/init",formation_msg, callback)
        talker()
        rospy.spin()   

    except rospy.ROSInterruptException:
        pass

