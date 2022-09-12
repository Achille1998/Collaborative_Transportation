
from numpy import diag
from scipy import zeros
import rospy
from geometry_msgs.msg import PoseStamped, Twist
from math import cos, sin, atan2 , pi
import numpy as np
from tf.transformations import euler_from_quaternion
import casadi as ca
from square_move.msg import force_msg
from nav_msgs.msg import Odometry

#class Parameters
all_object =[]


#extra class parameters
N_threshold = 0.1
all_pos_bool = []
#classes

Area = np.array([[-2 , 2],[-3 , 2]])


class Box:

    def __init__(self,name,target_pos_wrt_object):
        self.name = name
        self.pos = []
        self.vel=[]
        self.past_pos = []
        self.ang = 0.0
        self.orientation = []
        self.init_ang = 0.0
        self.N = 0.0
        self.m = 0.25
        self.a = 0.4
        self.b = 0.05
        self.c = 0.06  
        self.start= False
        self.target_pos_wrt_object = target_pos_wrt_object
        self.target_pos = []
        self.error_pub = rospy.Publisher('/Arm0/error_pose', force_msg, queue_size=1)

    def sub(self):
        return rospy.Subscriber("/qualisys/"+self.name+"/odom",Odometry,self.callback)
        
    def pub(self):
            
        return rospy.Publisher('/Arm0/vel_ref', force_msg, queue_size=1)
      

    def compute_distance_to_obj(self,obj): #compute distance and direction wrt other objects
        d = max(((self.pos.pose.position.x - obj.pos.pose.position.x )**2 + (self.pos.pose.position.y - obj.pos.pose.position.y )**2 )**0.5 - obj.dim, self.min_distance)
        return d

    def compute_ang_to_obj(self,obj):
        theta = atan2((obj.pos.pose.position.y - self.pos.pose.position.y ), (obj.pos.pose.position.x - self.pos.pose.position.x ))
        return theta



    def callback(self,odom):
        
        self.pos = np.array([odom.pose.pose.position.x , odom.pose.pose.position.y , odom.pose.pose.position.z])
        
        quat = [odom.pose.pose.orientation.x , odom.pose.pose.orientation.y , odom.pose.pose.orientation.z , odom.pose.pose.orientation.w]
        self.orientation = np.array(euler_from_quaternion(quat))
        self.vel = np.array([odom.twist.twist.linear.x,
                            odom.twist.twist.linear.y,
                            odom.twist.twist.linear.z,
                            odom.twist.twist.angular.x,
                            odom.twist.twist.angular.y,
                            odom.twist.twist.angular.z])
        #print(self.orientation)
        Theta = euler_from_quaternion(quat)
        theta = Theta[2]
        self.N,self.ang = count_lap(self.N,theta,self.ang)
        
        self.start = True

#-----------------------------------------------
    #           Navigation MPC controller
    #-----------------------------------------------
    def MPC_controller(self):

        # Compute the next time step based on the state model encoded in function f
        def shift_timestep(step_horizon, t0, state_init, u, f):
            f_value = f(state_init, u[:, 0])
            next_state = ca.DM.full(state_init + (step_horizon * f_value))

            t0 = t0 + step_horizon
            u0 = ca.horzcat(
                u[:, 1:],
                ca.reshape(u[:, -1], -1, 1)
            )
            return t0, next_state, u0

        # MPC paramters
        step_horizon = 0.005         # time between steps in seconds
        N = 16                # number of look ahead steps

        #------------------------------
        #   Setting weight matrices
        #------------------------------
        # State matrix weights
        Q_x = 10
        Q_theta = 100
        # Input matrix weights
        R_x = 0.8
        R_theta = 40
        # Max allowed velocity
        v_max = 0.2
        
        # State symbolic variables
        x = ca.SX.sym('x')
        y = ca.SX.sym('y')
        z = ca.SX.sym('z')
        theta = ca.SX.sym('theta')
        gamma = ca.SX.sym('gamma')
        psi = ca.SX.sym('psi')

        states = ca.vertcat(
            x,
            y,
            z,
            theta,
            gamma,
            psi
        )
        n_states = states.numel()               # number of states

        # Control input symbolic variables
        F_e = ca.SX.sym('fe',6,1)
        
        controls = F_e
        n_controls = controls.numel()           # number of control inputs


        ################OPTI#######################

        opti=ca.Opti()

        # # Matrix containing all states over all time steps +1 (each column is a state vector)
        X = opti.variable( n_states, N + 1)
        # # Matrix containing all control actions over all time steps (each column is an action vector)
        U = opti.variable( n_controls, N)
        
        # Coloumn vector for storing initial state and target state
        P = opti.parameter(n_states + n_states)
        
        # State weights matrix 
        Q = ca.diagcat(Q_x, Q_x, Q_x, Q_theta, Q_theta,  Q_theta)
        # Control input weights matrix
        R = ca.diagcat(R_x, R_x, R_x, R_theta, R_theta, R_theta)
        #s = opti.variable(n_states, N+1)
        
        #---------------------------------------------------
        #          KINEMATICS MODEL
        #--------------------------------------------------


        # A zero matrix
        # B id matrix
        # Rosie0 kinematics definition
        #gravity =  - 9.81 * Ry(-states[4,0])[:,2]
        # gravity = -9.81 * np.array([[ca.sin(-states[4,0])* ca.cos(-states[3,0]) ],
        #                             [ca.sin(-states[3,0])],
        #                             [ca.cos(-states[4,0]) * ca.cos(-states[3,0])] ])
        F_kinematics = controls
        
        f = ca.Function('f', [states, controls], [F_kinematics])

        #-----------------------------------------------------
        #       Cost function and constraints definition
        #-----------------------------------------------------
        J_k = 0                         # cost function initialization
        g = X[:, 0] - P[:n_states]      # initial condition state constraints

        # Runge-Kutta discretization
        for k in range(N):
            x_k = X[:, k]               # discretized states
            u_k = U[:, k]               # discretized control inputs
            
            # Cost function discretization
            if k == n_states-1:
              J_k = J_k \
                + ca.mtimes(ca.mtimes(u_k.T, R), u_k)
            else :
                J_k = J_k \
                    + ca.mtimes(ca.mtimes((x_k - P[n_states:]).T, Q), (x_k - P[n_states:])) \
                    + ca.mtimes(ca.mtimes(u_k.T, R), u_k)
                

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
        opti.subject_to(ca.vec(U)<= v_max)
        opti.subject_to(ca.vec(U)>=-v_max)
        
        
        p_opts = {"expand":True,
                "verbose": False}
        s_opts = {"max_iter": 500,
                    "print_level":0
                  }
        opti.solver("ipopt",p_opts,s_opts)

        #--------------------------------
        #       Online MPC navigation
        #--------------------------------
        t0 = 0
        
        state_init = ca.DM(ca.horzcat(self.pos.reshape((1,3)),self.orientation.reshape((1,3)) ) ).reshape((6,1))
              # initial state
        self.target_pos = self.pos + ca.mtimes(Rz(self.orientation[2]),self.target_pos_wrt_object[0:3].reshape((3,1))).reshape((1,3))
        
        state_target = ca.DM(ca.horzcat(self.target_pos, self.target_pos_wrt_object[3:6].reshape(1,3) )).reshape((6,1))    # target state

        t = ca.DM(t0)

        u0 = ca.DM.zeros((n_controls, N))          # initial control for the receding-horizon N
        X0 = ca.repmat(state_init, 1, N+1)         # initial full state for the receding-horizon N
        
        mpc_iter = 0

        def check_end(vec,final_threshold):
            bool_val = True
            for i,tresh_old in enumerate(final_threshold):
                bool_val = bool_val and abs(vec[i])<tresh_old
                print(bool_val)
            return bool_val
        
        ####################################################
        ####               MPC ITERATIONS               ####
        ####################################################

        final_threshold =[0.01,0.01,0.01,0.02,0.02,0.02]       # stopping condition

        while  not check_end(state_init - state_target,final_threshold):
            print(self.target_pos_wrt_object)
            print("state target:",state_target)
            print("state_init :",state_init )
            p = ca.vertcat(
                state_init,    # current state
                state_target   # target state
            )

            # Optimization solver solution

            opti.set_initial(X,X0)
            opti.set_initial(U,u0)
            opti.set_value(P,p)


            i=0
            
            try :
                sol = opti.solve()
            except RuntimeError as e:
                print(e)
                print(opti.return_status())
                opti.debug.show_infeasibilities()
                quit()
            
            opti.return_status()
            
            opti.debug.value(X)
            X0 = sol.value(X) 
            sol_u = sol.value(U)    # optimal control input sequence
             
            t0, state_init, u0 = shift_timestep(step_horizon, t0, state_init, sol_u, f)     # shift receding-horizon

            X0 = ca.horzcat(
                X0[:, 1:],
                ca.reshape(X0[:, -1], -1, 1)
            )


            # Send velocity commands to rosie0
            force_cmd = force_msg()
            
            force_cmd.force = (u0[:,0].reshape((1,6))).full()[0]
            
            
            # Publish velocity message
            rate = rospy.Rate(100)
            pub= self.pub()
            while pub.get_num_connections() == 0 and not rospy.is_shutdown():
                rate.sleep()
            rospy.loginfo(force_cmd)
            pub.publish(force_cmd)
            
            print("error :",(state_target - state_init[:, 0]).full())
            mpc_iter += 1
            
            state_init[:, 0] = ca.DM(ca.horzcat(self.pos.reshape((1,3)),self.orientation.reshape((1,3)) ) )        # feedback current state
            error_stamp = force_msg()
            error_stamp.force = (state_target - state_init[:, 0]).full()
            self.error_pub.publish(error_stamp)
            rate.sleep()
        return True



def count_lap(N,theta,old_theta):
    
    theta = theta + 2*pi*N
    
    if old_theta - theta  > 2 * pi -N_threshold: #one rotation
        
        N = N+1
        print("plus one lap")
        theta = theta + 2*pi
        print(N,theta,old_theta)
        
        return N,theta

    elif old_theta - theta < - 2* pi + N_threshold:
        N = N-1
        print("minus one lap")
        theta = theta - 2*pi
        print(N,theta,old_theta)
        return N,theta
        
    return N,theta 
    

def Rx(theta):
    Rx = np.array([[1 , 0 , 0],[0 , ca.cos ( theta ) , - ca.sin ( theta )],[0 , ca.sin ( theta ) , ca.cos ( theta )]])
    return Rx
def Ry(theta):
    Ry = np.array([[ ca.cos ( theta ) ,0, ca.sin ( theta )],[0 , 1 , 0],[- ca.sin ( theta ) ,0, ca.cos ( theta )]])
    return Ry
def Rz(theta):
    Rz =  np.array([[ ca.cos ( theta ) , - ca.sin ( theta ) , 0],[ ca.sin ( theta ) , ca.cos ( theta ) , 0],[0 ,0, 1]])
    return Rz





   