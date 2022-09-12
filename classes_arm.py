from casadi import *

import numpy as np
import math as mt
import rospy

from trajectory_msgs.msg import JointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from sensor_msgs.msg import JointState
from tf.transformations import euler_from_quaternion
from control_msgs.msg import GripperCommand
from square_move.msg import force_msg,target_pos
from Include.classes_base import Object,all_object,all_pos_bool, get_min_angle

all_arms = []
    
class Arm:

    def __init__(self,name,verbose=False,pub_bool=True):
        self.q_home = []
        self.name = name
        self.q_t = [] #actual pose
        self.q_t_dot = [] #actual velocity
        self.q_d_t = [] # next pose
        self.torque_t = [] #actual torque
        self.past_torque = []
        self.EE_force = [] # forces on EE
        self.EE_pos = [] # EE pos
        self.EE_pos_wrt_WF = [] # base pos + EE_pos with rotation Rz
        self.J_t = [] # actual jacobian

        self.vel_reference = force_msg().force #reference in force
        self.target_position = np.array([0.62,0,0.23,0,0,0]).reshape(6,1) #target
        self.f_e_vec = np.zeros((6,10))  #vector collect forces to filter
        self.g_t =[] #gravity compensation + static friction
        
        #object base
        base_name = "rosie"+self.name[-1]
        if base_name == "rosie0":
            self.base = Object(base_name,"leader")
        if base_name == "rosie1" or base_name == "rosie2"  :
            self.base = Object(base_name,"follower")


        #control params
        self.T0 =0
        self.start = False #wait until receive feedback
        self.init = True
        self.vel_start = False #wait untile receive vel
        self.max_update = 1
        self.verbose = verbose
        self.count_wait_feedback = 0
        self.verbose_count = 0
        self.target_received_bool = True
        self.time_to_stop_vel = 0
        self.grav_comp_count = 0

        #params grav comp
       
        #self.gains = [50,70,50,50,50,50] # Kp gains of the arm
        self.gains = [5, 10, 10, 10, 10, 5]
        self.gains = [2, 2, 3, 1, 1, 0.5]
        self.masses = [ 1.01798644,  0.47528919,  0.92991397,  0.2926678 ]
        self.fs = np.array([0, 1.17328604, -0.70002927,  0.15061906, -0.06275442,0 ])
        

        #publishers
        if pub_bool:
            self.grav_comp_pub = rospy.Publisher("/"+self.name +'/grav_torque',JointTrajectoryPoint,queue_size=1)
            self.force_pub = rospy.Publisher("/"+self.name+'/EE_force', force_msg, queue_size=1)
            self.pub_joints =rospy.Publisher("/"+self.name+'/joint_waypoints', JointTrajectory, queue_size=1)
            self.EE_pos_pub =rospy.Publisher("/"+self.name+'/EE_pose', target_pos, queue_size=1)
            self.gripper_pub = rospy.Publisher("/gripper_node"+self.name[-1]+"/gripper_cmd", GripperCommand, queue_size = 1)
        
    
        ################## compute positions of Motors ############################################

        l = np.array([[0.325],[0],[0]])  #length arm
        b1 = np.array([[0],[-0.06],[0.065]])  #brachet length
        b2 = np.array([[0],[-0.055],[0.08]])  #brachet length
        b3 = np.array([[0.055],[0],[0.06]])  #brachet length
        grip_l = np.array([[.11],[0],[0]])  #gripper length
        base_level = np.array([[0],[0],[0.24]]) 

        q = SX.sym("q",6,1)

        Motors_pos = horzcat(
            base_level,
            MMtimes( ( Rz(q[0]), b1 ) ) ,
            MMtimes( ( Rz(q[0]), Rx(pi/2), Rz(q[1]), l ) ) ,
            MMtimes( ( Rz(q[0]), Rx(pi/2), Rz(q[1]-q[2]),l ) ) ,
            MMtimes( ( Rz(q[0]), Rx(pi/2), Rz(q[1]-q[2] + q[3] ), b2 ) ),
            MMtimes( ( Rz(q[0]), Rx(pi/2), Rz(q[1]-q[2] + q[3] ), Rx(pi/2), Rz(q[4]), b3 )) ,
            MMtimes( ( Rz(q[0]), Rx(pi/2), Rz(q[1]-q[2] + q[3] ), Rx(pi/2), Rz(q[4]), Rx(-pi), grip_l )) )

        for i in range(Motors_pos.size(2)):
            if i >0:
                Motors_pos[:,i] = Motors_pos[:,i] + Motors_pos[:,i-1]
        #################################################################################
        ################## compute EE_pose and Jac functions #######################

        e_p = Motors_pos[:,-1]
        e_o = SX(np.array([[q[5]],[-q[1]+q[2]-q[3] ],[q[0]-q[4]]]))

        k_e = vertcat(e_p,e_o)
        J = jacobian(k_e,q)

        # save Functions for Jacb and EE pose
        self.F_J = Function("F_J",[q],[J])
        self.F_k_e = Function("F_k_e",[q],[k_e])

        ##################################################################################
        ############## grav compensation #########################################

        g_tau = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        j=0
        for i in range(Motors_pos.size(2)):
            
            if i >1 and i < (Motors_pos.size(2)) and i!=5:
                g_tau = g_tau + 9.81* gradient(Motors_pos[2,i],q)*self.masses[j]
                j = j+1
        # save function for gravity compensation

        self.F_g = Function("F_g",[q],[g_tau])

        #############################################################################
        ###################others initializations ############################
        self.gravity_torque = JointTrajectoryPoint()
        self.gravity_torque.positions = [0,0,0,0,0,0]
        self.gravity_torque.velocities = [0,0,0,0,0,0]
        self.gravity_torque.accelerations = [0,0,0,0,0,0]

        #############################################################
        #save all arms list
        global all_arms
        all_arms.append(self)


    #Subscribers
    def subscribers(self):

        rospy.Subscriber("/"+self.name+"/joint_states",JointState,self.update_all,queue_size=1)#########
        
        rospy.Subscriber("/"+self.name+'/vel_ref',force_msg,self.save_vel_ref,queue_size=100)
        
        rospy.Subscriber("/"+self.name+'/target_pos',target_pos,self.save_target_pos)

        self.base.sub()

    def set_initial_orientation(self,box_ang):
        
        if self.base.type == "leader":
            return get_min_angle(box_ang-self.base.orientation[2]+pi)
        else:
            return get_min_angle(box_ang-self.base.orientation[2])
        
    def gravity_compensator_mode(self,initial_pos,max_count_without_movement=300): 
        self.grav_comp_count += 1    
        self.past_torque = self.torque_t
        past_qt = self.q_t
        past_EE_pos = self.EE_pos
        rospy.sleep(0.01)
        error_xy = [self.EE_pos[0]- past_EE_pos[0] ,self.EE_pos[1]- past_EE_pos[1] ]
        rate=rospy.Rate(300)

        #if (np.abs(self.torque_t-self.past_torque)>0.2).any():
        if (np.abs(error_xy)>0.01).any():
            print("moving "+ self.name)
            print(self.target_position,error_xy)
            self.target_position[0] += error_xy[0]
            self.target_position[1] += error_xy[1]
            self.target_position.reshape((6,1))
            self.grav_comp_count = 0
        print(self.grav_comp_count)
        # if self.grav_comp_count == max_count_without_movement:
        #     self.grav_comp_count =0
        #     self.target_position = initial_pos
        

    def compute_q_d_t(self):

        if self.start:

            #check time
            if self.vel_start and rospy.Time.now().to_sec()-self.time_to_stop_vel >0.2:
                print("no more vel ref")
                self.vel_start = False

            if self.vel_start :
                
                ## Velocity controllee
                T1 = rospy.Time.now().to_sec()
                self.target_position =  self.target_position + self.vel_reference.reshape((6,1)) * (T1-self.T0)
                self.T0 = T1
                error_x = (self.target_position).reshape((6,1)) - self.EE_pos #error to mantain pos

            else:
                #position cart error
                error_x = self.target_position.reshape((6,1)) - self.EE_pos

            #position joint error

            J_t = self.F_J(self.q_t)
            update =  mtimes(pinv(J_t), error_x.reshape((6,1))).reshape((1,6))
            self.q_d_t = self.q_t + self.saturation(update) #+ gravity_comp
                 
            return True
            
        else:
            self.count_wait_feedback = self.count_wait_feedback +1
            if self.count_wait_feedback == 500000:
                print(" ----------" + self.name+ " -  Waiting for - FEEDBACK --------")
                self.count_wait_feedback = 0

            return False

    def create_msg(self):

        msg=JointTrajectory()
        
        p = JointTrajectoryPoint()
        
        for i in range(np.size(self.q_t,1)):
            
            p.positions.append(self.q_d_t[:,i])
            p.velocities.append(0)
            p.accelerations.append(0)
        p.time_from_start = rospy.Duration.from_sec(1.5)
        
        msg.points.append(p)

        return msg 
    
    def save_vel_ref(self,vel):
        self.target_received_bool = False

        #from box fram to robot frame
        big_Rz = horzcat( vertcat( Rz(-self.base.ang), np.zeros((3,3)) ), vertcat( np.zeros((3,3)), Rz(-self.base.ang) ) )

        self.vel_reference = (mtimes(big_Rz,np.array(vel.force).reshape((6,1)))).reshape((1,6))
        self.vel_reference[3:6] = [0,0,0]
        self.time_to_stop_vel = rospy.Time.now().to_sec()
        
        if self.vel_start == False :
            self.T0 = rospy.Time.now().to_sec() #initi time for integral action
            self.vel_start = True

        
    def save_target_pos(self,pos):
        
        if self.target_received_bool: # save data only if velref = 0
            self.target_position = np.array(pos.pos).reshape((6,1))
        

    def update_all(self,joints):

        #update joints pose, torques, compute EE pose Jacb and forces

        self.torque_t = np.array(joints.effort).reshape((1,6))
        # #torque_t[:,2] = -torque_t[:,2] #reversed motor
        self.q_t_dot = np.array(joints.velocity).reshape((1,6))
        self.q_t = np.array(joints.position).reshape((1,6))

        self.EE_pos = self.F_k_e(self.q_t)
        
        if self.base.pos_bool:
            self.EE_pos_wrt_WF = mtimes(Rz(self.base.ang),self.EE_pos[0:3,0]).reshape((1,3)) + (self.base.pos).reshape((1,3))
        
        # self.J_t = self.F_J(self.q_t)

        # self.g_t = self.F_g(self.q_t) + self.fs.reshape((6,1))

        # self.gravity_torque.effort= self.g_t.full()
                                    
        
        # compute force EE with filter
                #shifitg array
        # for i in range(np.size(self.f_e_vec,1)-1):
            
        #     self.f_e_vec[:,i]=self.f_e_vec[:,i+1]
        #     #stack new value
        # self.f_e_vec[:,-1] = mtimes(pinv(self.J_t.T), (self.torque_t.reshape((6,1))- self.g_t) ).reshape((1,6))
        
        # self.EE_force = np.mean(self.f_e_vec,1)

        # self.EE_force[0:3] = mtimes(Rz(-self.EE_pos[5]),self.EE_force[0:3].T).reshape((1,3))
        # update control param
        self.start = True
        
        self.print_verbose()

    def set_gripper(self,open_close):
        rate = rospy.Rate(50)
        gripper_cmd = GripperCommand()
        if open_close == "open":
            print("opening gripper on "+self.name)
            if self.name == "Arm2":
                gripper_cmd.position = 0.07
            else:
                gripper_cmd.position = 0.05
        if open_close == "close":
            
            print("closing gripper on "+self.name)
            if self.name == "Arm2":
                gripper_cmd.position = 0.02
            if self.name == "Arm0":
                gripper_cmd.position = 0.01
            else:  
                gripper_cmd.position = 0.0
        gripper_cmd.max_effort = 5
        while self.gripper_pub.get_num_connections() == 0:
                rate.sleep()
        self.gripper_pub.publish(gripper_cmd)


    def saturation (self,update):
        for i in range(update.numel()):
            if update[i] <0:
                update[i]= max(-self.max_update, update[i])
                if update[i]< -self.max_update: print("stauration - ")
            else:
                update[i] =min(self.max_update,update[i])
                if update[i]> self.max_update: print("stauration + ")
                
        return update

    def print_verbose(self):
        if  self.verbose: #print
            self.verbose_count = self.verbose_count +1
            if self.verbose_count == 100 :
                self.verbose_count = 0
                print("####################################################################")
                print("                           "+self.name)
                print("####################################################################")
                print("EE pos : ")
                print(self.EE_pos)
                print(self.q_t)
                print("EE pos error: ")
                print(self.target_position.reshape((6,1)) - self.EE_pos)
                print("joints pos : ")
                joint_error = self.q_d_t-self.q_t
                print(self.q_t)
                print("EE force : ")
                print(self.EE_force)
                print("torque : ")
                print(self.torque_t)
                print("gravity compensation : ")
                print(self.g_t)
                if self.vel_start:
                    print("vel ref :")
                    print( self.vel_reference)
                else:
                    print("no vel ref")

        rate = rospy.Rate(50)
        # msg = force_msg()
        #msg.force = self.EE_force
        #self.grav_comp_pub.publish(self.gravity_torque)
        #self.force_pub.publish(msg)
        if hasattr(self,'EE_pos_pub'):
            self.EE_pos_pub.publish(self.EE_pos.full())
        rate.sleep()
                


def Rx(theta):
    Rx = np.array([[1 , 0 , 0],[0 , cos ( theta ) , - sin ( theta )],[0 , sin ( theta ) , cos ( theta )]])
    return Rx
def Ry(theta):
    Ry = np.array([[ cos ( theta ) ,0, sin ( theta )],[0 , 1 , 0],[- sin ( theta ) ,0, cos ( theta )]])
    return SX(Ry)
def Rz(theta):
    Rz =  np.array([[ cos ( theta ) , - sin ( theta ) , 0],[ sin ( theta ) , cos ( theta ) , 0],[0 ,0, 1]])
    return Rz



def MMtimes(U):
    M = U[0]
    i=0
    for u in U:
        if i >0:
            M = mtimes(M,u)
        i = i+1
    return M

