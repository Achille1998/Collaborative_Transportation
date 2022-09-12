#!/usr/bin/env python

from casadi import *
import numpy as np
import rospy
from Include.classes_arm import *
from square_move.msg import formation_msg,force_msg
from Include.classes_base import Object

########## Initialization #############
Arm0 = Arm("Arm0",verbose=False)
Arm1 = Arm("Arm1",verbose=False)
Arm2 = Arm("Arm2",verbose=False)
Big_box = Object("Big_box","target")

############ Params ###########
form_msg = "move_arms"
gripper_status = [False,False,False]
Human_mode = False

def talker():
    start_positions = []
    start_human = True
    start = True
    msg = formation_msg() # control msg 
    rate = rospy.Rate(300)
    start_moving = False
    pub_formation = rospy.Publisher("/Formation/init",formation_msg, queue_size=10)

    while not rospy.is_shutdown():
        # arm orientation 
        for arm in all_arms:   
            arm.target_position[5,0] = arm.set_initial_orientation(Big_box.orientation[2]) #keep sending comands in orientation
           
        if all([arm.base.pos_bool for arm in all_arms]) and Big_box.pos_bool : #wait for orientation of the basis and object
            # open grippers , save initial time and set start position
            if start: 
                for arm in all_arms:
                    arm.set_gripper("open")

                Arm0.target_position = np.array([0.49,-0.01,0.235,0,0,arm.set_initial_orientation(Big_box.orientation[2])]).reshape(6,1)
                Arm1.target_position = np.array([0.46, 0.03,0.22,0,0,arm.set_initial_orientation(Big_box.orientation[2])]).reshape(6,1)
              
                Arm2.target_position = np.array([0.46,-0.06,0.24,0,0,arm.set_initial_orientation(Big_box.orientation[2])]).reshape(6,1)
           
                T0 = rospy.Time.now().to_sec()
                start = False

            if all([arm.compute_q_d_t() for arm in all_arms])  : #wait for feedback and compute cmd for joints

                # publish joints message for all arms
                for arm in all_arms:
                    arm.pub_joints.publish( arm.create_msg() )
                    
                T1 = rospy.Time.now().to_sec()

                if T1-T0 >3 and T1-T0 <3.5  : #wait 3s and send msg before moving bases
                    
                    msg.init = "move_base"
                    pub_formation.publish(msg)
                    rate.sleep()
                else:
                    # if formation error is 0
                    for i,arm in enumerate(all_arms):
                        if form_msg == "close_gripper_rosie"+arm.name[-1]:
                            gripper_status[i] = True
                    if all(gripper_status) and not(start_moving):
                        
                        # push
                        disp = np.array([0.04,0,0]).reshape((3,1))
                        for i,arm in enumerate(all_arms):
                            if arm.name == "Arm0":
                                arm.target_position += vertcat(mtimes(Rz(-arm.base.ang),-disp),0,0,0).full()
                            else:
                                arm.target_position += vertcat(mtimes(Rz(-arm.base.ang),disp),0,0,0).full()
                            arm.compute_q_d_t()
                            arm.pub_joints.publish( arm.create_msg() )
                            rate.sleep()
                            
                        rospy.sleep(2)

                        # close grippers
                        for arm in all_arms:
                            arm.set_gripper("close")
                            rate.sleep()

                        rospy.sleep(2)
                        
                        T1 = rospy.Time.now().to_sec()
                        # send msg that gripper are closed
                        while rospy.Time.now().to_sec()-T1 <1:
                            msg.init = "gripper_closed"
                            pub_formation.publish(msg)
                            rate.sleep()

                        start_moving = True

                    

                    elif not(start_moving):
                        print("Gripper Status : ",gripper_status)
                        rospy.sleep(0.5)

                if Human_mode: #not tested
                    #save all pos
                    if start_human:
                        for i,arm in enumerate(all_arms):
                            start_positions.append(arm.target_position)
                        start_human = False
                    for i,arm in enumerate(all_arms):
                        arm.gravity_compensator_mode(start_positions[i])
                     
     
                        
            rate.sleep() # sleep to wait joints value
        else:
            print("waiting for bases")
            rate.sleep()

def callback(msg):
    global form_msg
    global Human_mode
    form_msg = msg.init
    if form_msg == "start_Human_mode":
        Human_mode = True
    if form_msg == "stop_Human_mode":
        Human_mode = False

if __name__ == '__main__':
    try:
        rospy.init_node('Arms_node', anonymous=True)
        Big_box.sub()
        for arm in all_arms:
            arm.subscribers()
        rospy.Subscriber("/Formation/init",formation_msg, callback,queue_size=100)
        talker()
        
    except rospy.ROSInterruptException:
        pass


