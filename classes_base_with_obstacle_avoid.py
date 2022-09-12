


from casadi import norm_2
import rospy
from geometry_msgs.msg import PoseStamped, Twist
from math import cos, sin, atan2 , pi,sqrt
import numpy as np
from tf.transformations import euler_from_quaternion
import casadi as ca
from square_move.msg import formation_msg,target_pos

#class Parameters
all_object =[]


#extra class parameters
N_threshold = 0.0001
max_v = 0.05
max_w = 0.4 
all_pos_bool = []
#classes

Area = np.array([[-2 , 2],[-3 , 2]])


class Object:

    def __init__(self,name,type,pos_wrt_target=[0,0]):
        self.name = name
        self.pos = []
        self.pos_error = [0,0]
        self.ang_error = [0,0]
        self.target_pos = []
        self.Arm_EE_pos = np.zeros((2,1))
        self.pos_wrt_target = pos_wrt_target
        self.ang = 0.0
        self.orientation = np.array([0.0,0.0,0.0])
        self.init_ang = 0.0

        self.type = type
        self.object_grasped = False

        #control params
        self.start_target = False

        all_object.append(self)
        self.start = False  
        self.initialization = False
        self.pos_bool = False
        all_pos_bool.append(False)

        #############PUBLISHER#############

        if self.type == "follower" or self.type == "leader":
            self.pub_vel = rospy.Publisher("/"+self.name+"/cmd_vel", Twist, queue_size=1)
            self.pub_formation = rospy.Publisher("/Formation/init",formation_msg, queue_size=1)
            #self.pub_target_pos = rospy.Publisher("/"+self.name+"/target_pos",target_pos, queue_size=1)

     #############SUBSCRIBERS#############

    def sub(self):
        rospy.Subscriber("/qualisys/"+self.name+"/pose",PoseStamped,self.save_pos)

    def sub_formation_init(self):
        rospy.Subscriber("/Formation/init",formation_msg, self.callback)

    def sub_target_pos(self):
        rospy.Subscriber("/"+self.name+"/target_pos",target_pos,self.save_target_pos)
    
    def sub_arm_pos(self):
        rospy.Subscriber("/Arm"+self.name[-1]+'/EE_pose', target_pos, self.save_arm_pos)

    #############CALLBACKS#############
    def save_pos(self,pos):
        self.pos = np.array([pos.pose.position.x , pos.pose.position.y, 0.0])
        self.pos_bool = True

        if self.type == "leader" or self.type == "follower" or self.type == "target" :
            quat = [pos.pose.orientation.x , pos.pose.orientation.y , pos.pose.orientation.z , pos.pose.orientation.w]
            self.orientation = euler_from_quaternion(quat)
            self.ang = self.orientation[2]

    def callback(self,msg):
        if msg.init == "move_base":
            self.start_target = True
        if msg.init == "stop_bases":
            self.start_target = False
        if  msg.init == "gripper_closed":
            self.object_grasped = True

    def save_arm_pos(self,pos):
        self.Arm_EE_pos = ca.mtimes(Rz(self.ang),pos.pos[0:3])[0:2,:]

    def save_target_pos(self,pos):
        self.target_pos = pos.pos
        self.pos_error = np.array([pos.pos[0] - self.pos[0],pos.pos[1] - self.pos[1], pos.pos[2] - self.ang ])
    
    def compute_ang_to_obj(self,obj):
        theta = atan2((obj.pos[1] - self.pos[1] ), (obj.pos[0] - self.pos[0] ))
        return theta

    def save_initial_pos(self,obj):

        print("initial ang and diplacement setted")
        self.init_ang = obj.ang
        pos_wrt_target = self.pos - obj.pos
        print(pos_wrt_target)
        self.pos_wrt_target = ca.mtimes(Rz(-obj.ang) ,pos_wrt_target).reshape((1,3))[0:2] #save init pos wrt box fram
        print(self.pos_wrt_target)

    
    def compute_target_pos(self,obj):
        if obj.pos_bool:

            if  self.initialization: # after grasping object don't followe obj angle

                if self.object_grasped:
                    self.initialization =False
                    self.save_initial_pos(obj)                     
                else:
                
                    pos_wrt_target_WF = np.array([[self.pos_wrt_target[0]],[self.pos_wrt_target[1]],[0]]) #after initial pos
                    pos_wrt_target = ca.mtimes(Rz(obj.ang) ,pos_wrt_target_WF).reshape((1,3)) #compute pos to box wrt W frame
            else:
               
                pos_wrt_target_WF = np.array([[self.pos_wrt_target[0]],[self.pos_wrt_target[1]],[0]]) #after initial pos
                pos_wrt_target = ca.mtimes(Rz(self.init_ang) ,pos_wrt_target_WF).reshape((1,3)) #compute pos to box wrt W frame

            # target ang    
            theta_target = self.compute_ang_to_obj(obj)

            if abs(theta_target-self.ang)>=pi:
                theta_target = self.ang + get_min_angle(theta_target-self.ang) 

            target_pos_box = ca.horzcat(obj.pos[0:2]+ pos_wrt_target[0:2] , theta_target ).full()[0]

                # #publish
                # msg = target_pos
                # rate = rospy.Rate(10)
                # msg.pos = target_pos_box
                # self.pub_target_pos.publish(msg)
                # rate.sleep()
            
            return  target_pos_box

        else:
            print("no box pos")
            return [0,0,0]

    #-----------------------------------------------
    #           Navigation MPC controller
    #-----------------------------------------------
    def MPC_controller(self,box,formation_L,rosies):

        # MPC paramters
        step_horizon = 0.1         # time between steps in seconds
        N = 50
        d =  formation_L /sqrt(3)             # number of look ahead steps

        Q_x = 100
        Q_y = 100
        Q_theta = 1000
        Q_d=10000
        Q_o = 1000
        # Input matrix weights
        R1 = 20
        R2 = 20
        R3 = 100 
       
        # State symbolic variables
        x = ca.SX.sym('x')
        y = ca.SX.sym('y')
        theta = ca.SX.sym('theta')

        states = ca.vertcat(x,y,theta)
        n_states = states.numel()               # number of states

        # Control input symbolic variables
        vx_b = ca.SX.sym('vx_b')
        vy_b = ca.SX.sym('vy_b')
        w_b = ca.SX.sym('w_b')

        controls = ca.vertcat(vx_b,vy_b,w_b)
        n_controls = controls.numel()           # number of control inputs

        opti=ca.Opti()

        X = opti.variable( n_states, N + 1)

        U = opti.variable( n_controls, N)

        Q = ca.diagcat(Q_x, Q_y, Q_theta)

        R = ca.diagcat(R1, R2, R3)

        # # Rotation matrix from body to world frame
        R_B2W = ca.vertcat(
                    ca.horzcat(ca.cos(theta), -ca.sin(theta), 0),
                    ca.horzcat(ca.sin(theta),  ca.cos(theta), 0),
                    ca.horzcat(         0,           0, 1)
                )

        # Rosie0 kinematics definition
        F_kinematics = ca.mtimes(R_B2W, controls)
        
        f = ca.Function('f', [states, controls], [F_kinematics])

        #-----------------------------------------------------
        #       Cost function and constraints definition
        #-----------------------------------------------------
        #update target_state
        state_target = ca.DM(self.compute_target_pos(box)) 
        state_init = ca.DM([self.pos[0], self.pos[1], self.ang])   

        if box.pos_bool:
            box_pos = box.pos[0:2].reshape((2,1))

        J_k = 0                         # cost function initialization
        g = X[:, 0] - state_init     # initial condition state constraints
        
        # Runge-Kutta discretization
        for k in range(N):#for each state
            x_k = X[:, k]               # discretized states
            u_k = U[:, k]               # discretized control inputs

            # Cost function discretization
            J_k = J_k \
                + ca.mtimes(ca.mtimes((x_k - state_target).T, Q), (x_k - state_target)) \
                + ca.mtimes(ca.mtimes(u_k.T, R), u_k) \
                + Q_d*(ca.norm_2(x_k[0:2,:] - box_pos) - d)**2

            for rosie in rosies:
                if rosie.name != self.name and rosie.pos_bool :
                    obj_pos = rosie.pos[0:2].reshape((2,1))
                    J_k += Q_o/ca.norm_2(x_k[0:2,:]-obj_pos )**2     
                

            # Next discretized state
            x_k_next = X[:, k+1]
            k1 = f(x_k, u_k)
            k2 = f(x_k + step_horizon/2*k1, u_k)
            k3 = f(x_k + step_horizon/2*k2, u_k)
            k4 = f(x_k + step_horizon * k3, u_k)
            x_k_next_RK4 = x_k + (step_horizon / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            g = ca.vertcat(g, x_k_next - x_k_next_RK4) # costraint on possibile state that can be reach
            
        opti.minimize(J_k)
        opti.subject_to(opti.bounded(-0.001, ca.vec(g) , 0.001))

        ##Costraints##

        #max vel
        opti.subject_to(ca.vec(U[0:2,:])<=  max_v)
        opti.subject_to(ca.vec(U[0:2,:])>= -max_v)
        opti.subject_to(ca.vec(U[2,:])<=  max_w)
        opti.subject_to(ca.vec(U[2,:])>= -max_w)
        #walls
        opti.subject_to(opti.bounded(Area[0,0],ca.vec(X[0,:]),Area[0,1]))
        opti.subject_to(opti.bounded(Area[1,0],ca.vec(X[1,:]),Area[1,1]))

        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
        opti.solver("ipopt",opts)

        if self.start_target :
   
            u0 = ca.DM.zeros((n_controls, N))          # initial control for the receding-horizon N
            X0 = ca.repmat(state_init, 1, N+1)         # initial full state for the receding-horizon N
            #compute error
            self.pos_error = np.array([state_target[0] - self.pos[0], state_target[1] - self.pos[1], state_target[2] - self.ang ])


            p = ca.vertcat(
                state_init,    # current state
                state_target   # target state
            )
            box_pos = np.zeros((2,1))
            if box.pos_bool:
                box_pos = box.pos[0:2].reshape((2,1))

            # Optimization solver solution

            opti.set_initial(X,X0)
            opti.set_initial(U,u0)

            sol = opti.solve()
  
            sol_u = sol.value(U)    # optimal control input sequence
             
            # Apply first optimal control input
            vx_b = sol_u[0, 1]
            vy_b = sol_u[1, 1]
            w_b = sol_u[2, 1]

            velMsg = Twist()
            rate = rospy.Rate(100)
            final_threshold = 3*1e-2      # stopping condition
            # Send velocity commands to rosie0
            print(self.name+" pos error :" ,ca.norm_2(self.pos_error))
            if (ca.norm_2(self.pos_error[0:2]) > final_threshold):
                
                velMsg.linear.x = vx_b
                velMsg.linear.y = vy_b
                velMsg.angular.z = w_b

            elif not(self.object_grasped):
                msg = formation_msg()
                msg.init = "close_gripper_"+self.name
                print("send message: close_gripper_"+self.name)
                self.pub_formation.publish(msg)
                rospy.sleep(np.random.rand())
            
                # velMsg.linear.x = 0
                # velMsg.linear.y = 0
                # velMsg.angular.z = 0
            
            # Publish velocity message
            
            
            while self.pub_vel.get_num_connections() == 0:
                rate.sleep()
            rospy.loginfo(velMsg)
            self.pub_vel.publish(velMsg)
            rate.sleep()
            
            
        return True
       
# Functions

def saturation (vel):
    global max_vel
    if vel <0:
        return max (-max_vel, vel)
    else:
        return min (max_vel,vel)

def get_min_angle(angle):
    if angle > pi:
        return angle - 2*pi
    if angle < -pi :
        return angle + 2*pi
    else:
        return angle
 

def Rx(theta):
    Rx = np.array([[1 , 0 , 0],[0 , cos ( theta ) , - sin ( theta )],[0 , sin ( theta ) , cos ( theta )]])
    return Rx
def Ry(theta):
    Ry = np.array([[ cos ( theta ) ,0, sin ( theta )],[0 , 1 , 0],[- sin ( theta ) ,0, cos ( theta )]])
    return Ry
def Rz(theta):
    Rz =  np.array([[ cos ( theta ) , - sin ( theta ) , 0],[ sin ( theta ) , cos ( theta ) , 0],[0 ,0, 1]])
    return Rz







   