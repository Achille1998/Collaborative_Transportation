
from operator import truediv
import rospy
import casadi as ca
import numpy as np
from tf.transformations import euler_from_quaternion
from square_move.msg import force_msg,displacement,formation_msg,target_pos
from geometry_msgs.msg import PoseStamped
from Include.classes_base import get_min_angle

#extra class parameters
N_threshold = 0.1
v_max = 0.3
w_max = 0.1
all_pos_bool = []
#classes

Area = np.array([[-2 , 2],[-3 , 2]])


class Box:

    def __init__(self,formation,all_arms,G=np.zeros((3,3))):
        self.pos = []
        self.vel=[]
        self.orientation = []
        self.displacement = []
        self.target_pos = []
        self.starting_pos = np.array([0,0,0])
        self.start_orientation = [0,0,0]
        self.all_arms = all_arms
        self.formation = formation #formation.pos = [x,y,theta(z)]
        self.G = G #grasping Matrix [a/ b/ c] vectors from center to grasp point
        self.xyz_angles = []
        #control params
        self.start_odom= False
        self.start_disp = False
        self.object_attached = True ############# change to False ##########
        self.time_to_stop_formation = 0

        self.vel_pub = []
        # self.target_pub = []
        ########## PUBLISHERs ###############
        self.pos_pub = rospy.Publisher('/box/real_displacement', displacement, queue_size=1)
        
        for arm in self.all_arms:
            self.vel_pub.append(rospy.Publisher('/'+arm.name+'/vel_ref', force_msg, queue_size=1))
            # self.target_pub.append(rospy.Publisher('/'+arm.name+'/target_pos', target_pos, queue_size=1))
    ############ SUBSCRIBERs ##################

    def sub(self): 
        #subscribers for bases and box
        rospy.Subscriber("/qualisys/"+self.formation.box.name+"/pose",PoseStamped,self.save_pos)
        for obj in self.formation.obj_members:
            obj.sub()  

        rospy.Subscriber("/box/displacement",displacement,self.save_disp)
        rospy.Subscriber("/Formation/init",formation_msg, self.save_form_msg)

        for arm in self.all_arms:
            arm.subscribers()
           

    # Compute grasp matrix         
    def compute_G(self):
        G=np.zeros((3,3))
        for i,arm in enumerate(self.all_arms):
            if arm.start:
                box_pos = np.array([self.pos[0],self.pos[1],self.pos[2] + 0.238]).reshape((1,3))
                G[:,i] =  arm.EE_pos_wrt_WF - box_pos
        
        return G

    # callbacks
    def save_form_msg(self,msg):
        if msg.init == "gripper_closed":
            self.object_attached = True


    def save_disp(self,disp):
        self.displacement = np.array(disp.displacement)
        
        if self.start_disp ==False and self.object_attached:
            #save starting pos/ang
            self.starting_pos = self.pos
            self.start_orientation = self.orientation
            print("position saved ")
            ############################## Compute G
            if len(self.formation.obj_members) != 0 :  
                self.G = self.compute_G()
                print("MAtrix G : ",self.G)
                rospy.set_param("G_Matrix",ca.vec(self.G).full().tolist())
            
            self.start_disp = True
        self.time_to_stop_formation = rospy.Time.now().to_sec()

    def save_pos(self,pos):
        
        #compute pos wrt initial pose
        if len(self.formation.obj_members) == 0 :    
            self.pos = np.array([pos.pose.position.x - self.starting_pos[0] ,
                                pos.pose.position.y - self.starting_pos[1] ,
                                pos.pose.position.z - self.starting_pos[2]])
        #compute pos wrt center formation
        else: 
            self.formation.compute_pos()
            #COMPUTE RELATIVE POSITION WRT FORMATION
            self.pos = np.array([pos.pose.position.x - self.formation.pos[0] ,
                                pos.pose.position.y - self.formation.pos[1],
                                pos.pose.position.z - 0.238]) #height of the box

        #compute ang wrt initial ang (in any case)

        quat = [pos.pose.orientation.x , pos.pose.orientation.y , pos.pose.orientation.z , pos.pose.orientation.w]
        
        self.xyz_angles = np.array(euler_from_quaternion(quat))
        self.orientation = ca.horzcat(self.xyz_angles[0],self.xyz_angles[1], get_min_angle( self.xyz_angles[2] - self.formation.pos[2] ) )

        # #publish pos object
        msg = displacement()
        msg.displacement = ca.horzcat(self.pos.reshape((1,3)),self.orientation.reshape((1,3))).full()[0]
    
        self.pos_pub.publish(msg)
        self.start_odom = True
    
    
  
       
        
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
        step_horizon = 0.01         # time between steps in seconds
        N = 4 # true:16                # number of look ahead steps

        #------------------------------
        #   Setting weight matrices
        #------------------------------
        # State matrix weights
        Q_x = 8
        Q_theta = 2
        #Q_G = 1
        # Input matrix weights
        R_x = 0.5
        R_theta = 4
        # Max allowed velocity
        
        
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
        n_states = states.numel()       # number of states

        # Control input symbolic variables
        F_e = ca.SX.sym('fe',6,1)
        
        controls = F_e
        n_controls = controls.numel()     # number of control inputs


        ############VARIABLE AND PARAMETERS OPTI##########

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
        
        EE_POS = opti.parameter(3,3)
        
        #---------------------------------------------------
        #          KINEMATICS MODEL
        #--------------------------------------------------
       
        F_kinematics = controls #velocity wrt body frame
        
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

                # for j,arm in enumerate(self.all_arms): #penalize position error between object and EE 
                #     J_k = J_k + Q_G *ca.norm_2(x_k[0:2].reshape((1,2)) + self.G[0:2,j].reshape((1,2)) - EE_POS[j,0:2])**2

                

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
        for i in range(U.shape[1]): #for each column
            for j,arm in enumerate(self.all_arms):
                opti.subject_to(ca.vec (U[0:3,i] ) <= v_max)
                opti.subject_to(ca.vec (U[0:3,i] )  > -v_max)
                
                opti.subject_to(ca.vec ( ca.cross(U[3:6,i],self.G[:,j]) ) <= w_max)
                opti.subject_to(ca.vec ( ca.cross(U[3:6,i],self.G[:,j]))  > -w_max)

                opti.subject_to(ca.vec (U[0:3,i]+ ca.cross(U[3:6,i],self.G[:,j]) ) <= v_max)
                opti.subject_to(ca.vec (U[0:3,i]+ ca.cross(U[3:6,i],self.G[:,j]))  > -v_max)
                
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
        

        opti.solver("ipopt",opts)

        #--------------------------------
        #       Online MPC navigation
        #--------------------------------
        t0 = 0
        # initial state
        state_init = ca.DM(ca.horzcat(self.pos.reshape((1,3)),self.orientation.reshape((1,3)) ) ).reshape((6,1))
        # target state
        state_target = ca.DM(self.displacement).reshape((6,1))    
        # EE_POS_value = np.zeros((3,3))
        t = ca.DM(t0)

        u0 = ca.DM.zeros((n_controls, N))          # initial control for the receding-horizon N
        X0 = ca.repmat(state_init, 1, N+1)         # initial full state for the receding-horizon N
        
        

        #ending condition
        def check_end(vec,final_threshold):
            bool_val = True
            for i,tresh_old in enumerate(final_threshold):
                #print("goal ",i," reached",abs(vec[i])<tresh_old)
                bool_val = bool_val and abs(vec[i])<tresh_old
               
            return bool_val
        
        ####################################################
        ####               MPC ITERATIONS               ####
        ####################################################

        final_threshold =[0.01,0.01,0.01,0.02,0.02,0.02]       # stopping condition
        
        self.time_to_stop_formation = rospy.Time.now().to_sec()
        rospy.loginfo("STARTING Displacement MPC")
        
        
        while  not check_end(state_init - state_target,final_threshold) and self.start_disp:

            #check for target_pos
            if rospy.Time.now().to_sec()-self.time_to_stop_formation >0.5:
                rospy.loginfo("no more target ref")
                self.start_disp = False
            
            ###### check for new references #######
            state_target = ca.DM(self.displacement).reshape((6,1)) 

            # print("state target:",state_target)
            # print("state_init :",state_init )
            p = ca.vertcat(
                state_init,    # current state
                ca.DM(self.displacement).reshape((6,1))   # target state updated online
            )

            # for i,arm in enumerate(self.all_arms):
            #     EE_POS_value[i,:] = arm.EE_pos_wrt_WF.reshape((1,3))    
            
            # Optimization solver solution

            opti.set_initial(X,X0)
            opti.set_initial(U,u0)
            opti.set_value(P,p)
            # opti.set_value(EE_POS,EE_POS_value)

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
            
            
            
            # Publish velocity message
            rate = rospy.Rate(300)
            for i,arm in enumerate(self.all_arms):
                v = u0[0:3,0].reshape((1,3)).full()[0]
                w = u0[3:6,0].reshape((1,3)).full()[0]
                pub = self.vel_pub[i]
                # WAINT UNTIL CONNECTIONS   
                while pub.get_num_connections() == 0 and not rospy.is_shutdown():
                    print("{} is not connected".format(arm.name))
                    rate.sleep()
                force_cmd = force_msg()
                tot_vel_WF =  (v + ca.cross(w,self.G[:,i]) )
                force_cmd.force = (ca.vertcat( tot_vel_WF.reshape((1,3)).full()[0], [0,0,0])).reshape((1,6)).full()[0]
                force_cmd.linear = v.reshape((1,3))[0]
                force_cmd.angular = ca.cross(w,self.G[:,i]).reshape((1,3)).full()[0]
                
                #rospy.loginfo(arm.name,force_cmd.force) 
                pub.publish(force_cmd)
                
                
                rate.sleep()
               

            #print("error :",(state_target - state_init[:, 0]).full())
            
            #self.compute_G() #compute G (in case of changes)
            state_init[:, 0] = ca.DM(ca.horzcat(self.pos.reshape((1,3)),self.orientation.reshape((1,3)) ) )        # feedback current state
            # error_stamp = force_msg()
            # error_stamp.force = (ca.DM(self.displacement).reshape((6,1)) - state_init[:, 0]).full()
            # self.error_pub.publish(error_stamp)
            # rate.sleep()
        return True

    

def Rx(theta):
    Rx = np.array([[1 , 0 , 0],[0 , ca.cos ( theta ) , - ca.sin ( theta )],[0 , ca.sin ( theta ) , ca.cos ( theta )]])
    return Rx
def Ry(theta):
    Ry = np.array([[ ca.cos ( theta ) ,0, ca.sin ( theta )],[0 , 1 , 0],[- ca.sin ( theta ) ,0, ca.cos ( theta )]])
    return Ry
def Rz(theta):
    Rz =  np.array([[ ca.cos ( theta ) , - ca.sin ( theta ) , 0],[ ca.sin ( theta ) , ca.cos ( theta ) , 0],[0 ,0, 1]])
    return Rz





   