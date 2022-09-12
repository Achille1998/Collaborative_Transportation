
import rospy
from geometry_msgs.msg import PoseStamped, Twist
from square_move.msg import displacement,target_pos
from math import cos, sin, atan2 , pi
import numpy as np
from Include.classes_base import get_min_angle
from tf.transformations import euler_from_quaternion
import casadi as ca
from square_move.msg import formation_msg


Area = np.array([[-2 , 2],[-3 , 2]])

class Formation:

    def __init__(self,obj_members,box,weights=[0,0,0,0,0,0],max_lin_disp=0.05,max_ang_disp = pi/16,pub_bool=True):
        self.pos = [0.0,0.0]
        self.obj_members = obj_members
        self.box = box
        self.max_lin_disp = max_lin_disp
        self.max_ang_disp = max_ang_disp
        self.kp = [1,1,1]
        self.ang = 0.0
        self.target_pos = []
        self.weights = weights
        #control params
        self.start = False
        self.pos_bool = False
        self.time_to_stop_formation = 0
        self.pub_bool = pub_bool
        

        #publisher
        self.pub_disp = rospy.Publisher("/box/displacement", displacement, queue_size=1)
        self.pub_formation_pos = rospy.Publisher("/Formation/pose",formation_msg, queue_size=1)
        self.pub_formation = rospy.Publisher("/Formation/init",formation_msg, queue_size=10)
    def sub(self):
        rospy.Subscriber("/box/target_pos",target_pos,self.callback)
        #rosie positions
        for obj in self.obj_members:
            obj.sub()  
        self.box.sub()

    def compute_pos(self): 
        x =0
        y =0
        theta = 0  #reset rotation to 0 
        for i,obj in enumerate(self.obj_members):
            if obj.pos_bool:
                x =  x  + obj.pos[0]
                y =  y  + obj.pos[1]
                ang_wrt_center = get_min_angle( atan2((obj.pos[1] - self.pos[1] ), (obj.pos[0] - self.pos[0] )) -2*i*pi/3 )
                theta += ang_wrt_center
                
        self.pos = [x/3, y/3, get_min_angle(theta/3)]
        if self.pub_bool:
            msg=formation_msg()
            msg.pos = self.pos
            self.pub_formation_pos.publish(msg)
        
    def callback(self,target):
        
        self.target_pos = target.pos
        self.time_to_stop_formation = rospy.Time.now().to_sec()
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
        step_horizon = 0.3        # time between steps in seconds
        N = 12                  # number of look ahead steps

        #------------------------------
        #   Setting weight matrices
        #------------------------------
        # State matrix weights
        Q_x = self.weights[0]
        Q_y = self.weights[1]
        Q_theta = self.weights[2]

        # Input matrix weights
        R1 = self.weights[3]
        R2 = self.weights[4]
        R3 = self.weights[5]

        
        # Initial state specifications
        self.compute_pos()
        x_init = self.box.pos[0]
        y_init = self.box.pos[1]
        theta_init = self.ang


        # Max allowed velocity
   
        
        # State symbolic variables
        x = ca.SX.sym('x')
        y = ca.SX.sym('y')
        theta = ca.SX.sym('theta')

        states = ca.vertcat(
            x,
            y,
            theta
        )
        n_states = states.numel()               # number of states

        # Control input symbolic variables
        vx_b = ca.SX.sym('vx_b')
        vy_b = ca.SX.sym('vy_b')
        w_b = ca.SX.sym('w_b')

        controls = ca.vertcat(
            vx_b,
            vy_b,
            w_b,

        )
        n_controls = controls.numel()      # number of control inputs

        ################OPTI#######################

        opti=ca.Opti()

        # # Matrix containing all states over all time steps +1 (each column is a state vector)
        X = opti.variable( n_states, N + 1)
        # # Matrix containing all control actions over all time steps (each column is an action vector)
        U = opti.variable( n_controls, N)
        # Coloumn vector for storing initial state and target state
        P = opti.parameter(n_states + n_states)
        # POS = opti.parameter(len(all_object)-1,2) 
        # State weights matrix 
        Q = ca.diagcat(Q_x, Q_y, Q_theta)
        # Control input weights matrix
        R = ca.diagcat(R1, R2, R3)

        #---------------------------------------------------
        #           Formation KINEMATICS MODEL
        #--------------------------------------------------

        # # Rotation matrix from body to world frame
        R_B2W = ca.vertcat(
                    ca.horzcat(ca.cos(theta), -ca.sin(theta), 0),
                    ca.horzcat(ca.sin(theta),  ca.cos(theta), 0),
                    ca.horzcat(         0,           0, 1)
                )

        # Rosie0 kinematics definition
        F_kinematics = ca.mtimes(R_B2W,  controls)
        
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
            J_k = J_k \
                + ca.mtimes(ca.mtimes((x_k - P[n_states:]).T, Q), (x_k - P[n_states:])) \
                + ca.mtimes(ca.mtimes(u_k.T, R), u_k) \
                #+ 10000*ca.norm_2(s[:,k])

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
        opti.subject_to(opti.bounded(-0.001, ca.vec(g) , 0.0001))

        ##Costraints##

        #max vel
        opti.subject_to(ca.vec(U[0:2,:])<=  self.max_lin_disp)
        opti.subject_to(ca.vec(U[0:2,:])>=- self.max_lin_disp)
        opti.subject_to(ca.vec(U[2,:])<=  self.max_ang_disp)
        opti.subject_to(ca.vec(U[2,:])>=- self.max_ang_disp)
        #walls
        # opti.subject_to(opti.bounded(Area[0,0],ca.vec(X[0,:]),Area[0,1]))
        # opti.subject_to(opti.bounded(Area[1,0],ca.vec(X[1,:]),Area[1,1]))
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
        #       Online MPC navigation
        #--------------------------------
        t0 = 0
        state_init = ca.DM([x_init, y_init, theta_init])            # initial state
        state_target = ca.DM([self.target_pos[0], self.target_pos[1], self.target_pos[3]])    # target state
        t = ca.DM(t0)

        u0 = ca.DM.zeros((n_controls, N))          # initial control for the receding-horizon N
        X0 = ca.repmat(state_init, 1, N+1)         # initial full state for the receding-horizon N
        
        mpc_iter = 0
        
        
        ####################################################
        ####               MPC ITERATIONS               ####
        ####################################################

        final_threshold =5e-2      # stopping condition
        # while (ca.norm_2(state_init - state_target) > final_threshold):
        
        rospy.loginfo("STARTING FORMATION MPC")
        while not rospy.is_shutdown() and self.start: 
            #check for target_pos
            if rospy.Time.now().to_sec()-self.time_to_stop_formation >0.5:
                rospy.loginfo("no more target ref")
                self.start = False

            #change online the target
            state_target = ca.DM([self.target_pos[0], self.target_pos[1], self.target_pos[3]])
            z_target = self.target_pos[2]
            p = ca.vertcat(
                state_init,    # current state
                state_target   # target state
            )

            # Optimization solver solution

            opti.set_initial(X,X0)
            opti.set_initial(U,u0)
            opti.set_value(P,p)

            
            sol = opti.solve()
  
            X0 = sol.value(X) 
            sol_u = sol.value(U)    # optimal control input sequence
             
            t0, state_init, u0 = shift_timestep(step_horizon, t0, state_init, sol_u, f)     # shift receding-horizon

            X0 = ca.horzcat( #shifting
                X0[:, 1:],
                ca.reshape(X0[:, -1], -1, 1)
            )

            # Apply first optimal control input
            dx = sol_u[0, 1]
            dy = sol_u[1, 1]
            dw = sol_u[2, 1]
            
            # Send velocity commands to rosie0
            displacement_msg = displacement()
            msg = formation_msg()
            rate = rospy.Rate(50)
            if ca.norm_2(state_init-state_target) > final_threshold:

                displacement_msg.displacement = [dx, dy, z_target, 0, 0, 0]
                msg.init="move_base"
            else:
                displacement_msg.displacement = [0,0, z_target, 0, 0, 0]
                msg.init="stop_base"
            
            while self.pub_disp.get_num_connections() == 0:
                rate.sleep()
          
            self.pub_disp.publish(displacement_msg)
            
            self.pub_formation.publish(msg)
            rate.sleep()

            mpc_iter += 1
            self.compute_pos()
            state_init[:, 0] = [self.box.pos[0], self.box.pos[1], self.ang]         # feedback current state

        return True

    







   
