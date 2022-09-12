


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
max_v = 0.1
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
        self.box_ang = 0.0
        self.form_pos = []
        self.type = type
        self.object_grasped = False

        #control params
        self.start_target = False
        self.time_target_ang = 0

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
        rospy.Subscriber("/box/target_pos",target_pos,self.save_target_pos)
    
    def sub_arm_pos(self):
        rospy.Subscriber("/Arm"+self.name[-1]+'/EE_pose', target_pos, self.save_arm_pos)

    def sub_formation_pose(self):
        rospy.Subscriber("/Formation/pose",formation_msg, self.save_form_pose)


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
        if msg.init == "stop_bases" and self.object_grasped:
            self.start_target = False
        if  msg.init == "gripper_closed":
            self.object_grasped = True

    def save_form_pose(self,pos):
       self.form_pos = pos.pos

    def save_arm_pos(self,pos):
        self.Arm_EE_pos = ca.mtimes(Rz(self.ang),pos.pos[0:3])[0:2,:]

    def save_target_pos(self,pos):
        self.box_ang = pos.pos[3]
        self.time_target_ang = rospy.Time.now().to_sec()
    
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
                # if (rospy.Time.now().to_sec() - self.time_target_ang) > 0.5 and len(self.form_pos)>0:
                    
                #     print("manual control")

                #     self.box_ang = cut_treshold( obj.ang - self.form_pos[2] , 0.08 ) +  self.form_pos[2] 

                pos_wrt_target_WF = np.array([[self.pos_wrt_target[0]],[self.pos_wrt_target[1]],[0]]) #after initial pos
                pos_wrt_target = ca.mtimes(Rz(self.box_ang + self.init_ang) ,pos_wrt_target_WF).reshape((1,3)) #compute pos to box wrt W frame


            # target ang    
            theta_target = self.compute_ang_to_obj(obj)

            if abs(theta_target-self.ang)>=pi:
                theta_target = self.ang + get_min_angle(theta_target-self.ang) 

            target_pos_box = ca.horzcat(obj.pos[0:2]+ pos_wrt_target[0:2] , theta_target ).full()[0]
            # if len(self.form_pos)>1:
            #     if abs(self.box_ang - self.form_pos[2] )>0.2 : # block movements in xy with formation rotations
                
            #         target_pos_box = ca.horzcat(np.array(self.form_pos[0:2]).reshape((1,2)) + pos_wrt_target[0:2] , theta_target ).full()[0]
 
            return  target_pos_box

        else:
            print("no box pos")
            return [0,0,0]

    #-----------------------------------------------
    #           Navigation MPC controller
    #-----------------------------------------------
    def MPC_controller(self,box,formation_L):

        # Compute the next time step based on the state model encoded in function f
        def shift_timestep(step_horizon, state_init, u, f):
            f_value = f(state_init, u[:, 0])
            next_state = ca.DM.full(state_init + (step_horizon * f_value))

           
            u0 = ca.horzcat(
                u[:, 1:],
                ca.reshape(u[:, -1], -1, 1)
            )
            return  next_state, u0

        # MPC paramters
        step_horizon = 0.1         # time between steps in seconds
        N = 4            # number of look ahead steps
        d =  formation_L /sqrt(3)            

        #------------------------------
        #   Setting weight matrices
        #------------------------------
        # State matrix weights #max vel 0.05
        Q_x = 2000 #1000 
        Q_y = 2000 #1000
        Q_theta = 1000
        Q_d=0
        # Q_G_value = 0

        # Input matrix weights
        R1 = 200
        R2 = 200
        R3 = 100

        
        # Initial state specifications
        x_init = self.pos[0]
        y_init = self.pos[1]
        theta_init = self.ang

        theta_target = 0.0        
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

        ################OPTI#######################

        opti=ca.Opti()

        # # Matrix containing all states over all time steps +1 (each column is a state vector)
        X = opti.variable( n_states, N + 1)
        # # Matrix containing all control actions over all time steps (each column is an action vector)
        U = opti.variable( n_controls, N)
        # Coloumn vector for storing initial state and target state
        P = opti.parameter(n_states + n_states)
        
        
        box_POS = opti.parameter(2,1)
        # Q_G = opti.parameter(1)
         
        # State weights matrix 
        Q = ca.diagcat(Q_x, Q_y, Q_theta)
        # Control input weights matrix
        R = ca.diagcat(R1, R2, R3)
        #s = opti.variable(n_states, N+1)

        #---------------------------------------------------
        #           ROSIE0 KINEMATICS MODEL
        #--------------------------------------------------

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
        J_k = 0                         # cost function initialization
        g = X[:, 0] - P[:n_states]      # initial condition state constraints
        j = int(self.name[-1])
        # Runge-Kutta discretization
        for k in range(N):
            x_k = X[:, k]               # discretized states
            u_k = U[:, k]               # discretized control inputs

            # Cost function discretization
            J_k = J_k \
                + ca.mtimes(ca.mtimes((x_k - P[n_states:]).T, Q), (x_k - P[n_states:])) \
                + ca.mtimes(ca.mtimes(u_k.T, R), u_k) \
                + Q_d*(ca.norm_2(x_k[0:2,:] - box_POS) - d)**2
                # + Q_G *ca.norm_2(x_k[0:2,:] + G[0:2,j] - EE_POS )**2   
                

            # Next discretized state
            x_k_next = X[:, k+1]
            k1 = f(x_k, u_k)
            k2 = f(x_k + step_horizon/2*k1, u_k)
            k3 = f(x_k + step_horizon/2*k2, u_k)
            k4 = f(x_k + step_horizon * k3, u_k)
            x_k_next_RK4 = x_k + (step_horizon / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            g = ca.vertcat(g, x_k_next - x_k_next_RK4) # costraint on possibile state that can be reach
            
        #---------------------------------------
        #          Optimization specs OPTI
        #---------------------------------------
        
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
        #collisions
        # for i in range(len(all_object)-1):
        #     for j in range(N+1):
        #         opti.subject_to( (X[0,j]- POS[i,0])**2 + (X[1,j] - POS[i,1])**2 > 0.1 )
        
        # for i in range(len(all_object)-1):
        #     # for j in range(N+1):
        #         opti.subject_to( X[0,1]**2 + X[1,1]**2  > 0.01 )
   
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
        opti.solver("ipopt",opts)

        #--------------------------------
        #      INIZIALIZATION
        #--------------------------------
        state_init = ca.DM([x_init, y_init, theta_init])            # initial state
        G_value = np.zeros((3,3))
        u0 = ca.DM.zeros((n_controls, N))          # initial control for the receding-horizon N
        X0 = ca.repmat(state_init, 1, N+1)         # initial full state for the receding-horizon N
        
        ####################################################
        ####               MPC ITERATIONS               ####
        ####################################################

        rospy.loginfo("STARTING "+self.name+" MPC")

        while not rospy.is_shutdown() and self.start_target :

            #update target_state
            state_target = ca.DM(self.compute_target_pos(box)) 
            print(self.name,state_target)
            #compute error
            self.pos_error = np.array([state_target[0] - self.pos[0], state_target[1] - self.pos[1], state_target[2] - self.ang ])

            #update param
           
         
            # if rospy.has_param("G_Matrix"):
            #     G_value= np.array(rospy.get_param("G_Matrix"))
            #     print(G_value.reshape((3,3)))
            #     Q_G_value = 100

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
            opti.set_value(P,p)
            opti.set_value(box_POS,box_pos)
            
            
            # opti.set_value(Q_G,Q_G_value)
            try :
                sol = opti.solve()
            except RuntimeError as e:
                print(e)
                print(opti.return_status())
                opti.debug.show_infeasibilities()
                quit()

            X0 = sol.value(X) 
            sol_u = sol.value(U)    # optimal control input sequence
             
            state_init, u0 = shift_timestep(step_horizon, state_init, sol_u, f)     # shift receding-horizon

            X0 = ca.horzcat(
                X0[:, 1:],
                ca.reshape(X0[:, -1], -1, 1)
            )

            # Apply first optimal control input
            vx_b = u0[0, 0]
            vy_b = u0[1, 0]
            w_b = u0[2, 0]

            velMsg = Twist()
            rate = rospy.Rate(100)
            velMsg.linear.x = vx_b
            velMsg.linear.y = vy_b
            velMsg.angular.z = w_b
            final_threshold = 3*1e-2      # stopping condition
            # Send velocity commands to rosie0
            print(self.name+" pos error :" ,ca.norm_2(self.pos_error))

            if (ca.norm_2(self.pos_error[0:2]) < final_threshold) and not(self.object_grasped) :
                msg = formation_msg()
                msg.init = "close_gripper_"+self.name
                print("send message: close_gripper_"+self.name)
                self.pub_formation.publish(msg)
                rospy.sleep(np.random.rand())

            # Publish velocity message
            
            
            while self.pub_vel.get_num_connections() == 0:
                rate.sleep()
            #rospy.loginfo(velMsg)
            self.pub_vel.publish(velMsg)
            rate.sleep()
            
            state_init[:, 0] = [self.pos[0], self.pos[1], self.ang]         # feedback current state
            
        return True
       
# Functions

def saturation (vel,max=max_v):
    if vel <0:
        return max (-max, vel)
    else:
        return min (max,vel)

def cut_treshold(x,treshold):
    if abs(x) <= treshold:
        return 0
    if x > treshold:
        return x-treshold
    if x < treshold:
        return x + treshold

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







   
