from matplotlib import pyplot as plt
import numpy as np
import numba as nb
from scipy.spatial.transform import Rotation

class Sim:
    def __init__(self, model):
        self.model = model
    
    def rk4(self,f,x,u,dt):
        k1 = f(x,u)
        k2 = f(x + 0.5*dt*k1,u)
        k3 = f(x + 0.5*dt*k2,u)
        k4 = f(x + dt*k3,u)
        final = x + dt*(k1 + 2*k2 + 2*k3 + k4)/6
        return final
    
    def step(self,x,u):
        return self.rk4(self.model.dynamics,x,u,self.model.dt)
    

class QuadRotor:
    g = 9.81 #m/s^2
    m = 1#kg
    J = np.diag([.1,.1,.1])#inertia matrix, kg*m^2
    J_inv = np.linalg.inv(J)
    L = 0.1 #m
    dt = 0.01
    
    def __init__(self):
        self.x = np.zeros((13,1))
        self.u = np.zeros((4,1))

    @staticmethod
    def dynamics(x,u):
        # x = [p,v,q,w]*N u = [t,tau]*N in R^4

        N = x.shape[1]
        v = x[3:6]
        q = x[6:10]
        w = x[10:13]
        
        colective_thrust = u[0]
        tau = u[1:4]
        
        qw = q[0]
        qx = q[1]
        qy = q[2]
        qz = q[3]
        
        wx = w[0]
        wy = w[1]
        wz = w[2]
        
        x_dot = np.zeros((13,N))
        x_dot[0:3] = v
        v_dot = x_dot[3:6]
        v_dot[0] =  2*(qw*qy+qx*qz)*colective_thrust
        v_dot[1] =  2*(qy*qz-qw*qx)*colective_thrust
        v_dot[2] = (1-2*(qx**2+qy**2))*colective_thrust - QuadRotor.g

        q_dot = x_dot[6:10]
        q_dot[0] =  0.5*(-wx*qx - wy*qy - wz*qz)
        q_dot[1] =  0.5*(wx*qw + wz*qy - wy*qz)
        q_dot[2] =  0.5*(wy*qw - wz*qx + wx*qz)
        q_dot[3] =  0.5*(wz*qw + wy*qx - wx*qy)
        x_dot[10:13] = QuadRotor.J_inv@(tau - np.cross(w.T,np.dot(QuadRotor.J,w).T).T)
        return x_dot

class Trajectory:
    @staticmethod
    def circle(t,r,h):
        p = np.array([r*np.cos(t),r*np.sin(t),h]).T
        v = np.array([-r*np.sin(t),r*np.cos(t),0]).T
        a = np.array([-r*np.cos(t),-r*np.sin(t),0]).T
        q = np.array([1,0,0,0]).T
        w = np.array([0,0,0]).T
        return np.concatenate([p,v,q,w])
    
    @staticmethod
    def position(t,p):
        p = p
        v = np.array([0,0,0]).T
        q = np.array([1,0,0,0]).T
        w = np.array([0,0,0]).T
        return np.concatenate([p,v,q,w])
    

class Visualizer:
    def __init__(self,l = 0.25):
        # Visualize the quadrotor position, orientation and reference trajectory
        # and MPPI sample results
        self.xs = []
        self.x_ref = []
        self.x_mppi = []

        self.l = l
        
        plt.ion()
        fig = plt.figure()
        # for stopping simulation with the esc key.
        fig.canvas.mpl_connect('key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
        self.ax = fig.add_subplot(111, projection='3d')

    
    def update(self,x,x_ref,x_mppi):
        self.xs.append(x)
        self.x_ref = x_ref
        self.x_mppi = x_mppi
    
    def draw(self):
        p1 = np.array([self.l / 2, 0, 0]).T
        p2 = np.array([- self.l/ 2, 0, 0]).T
        p3 = np.array([0, self.l/ 2, 0]).T
        p4 = np.array([0, -self.l/ 2, 0]).T
        
        
        p = self.xs[-1][0:3,0]
        q = self.xs[-1][6:10,0]
        ax = self.ax
        
        rot = Rotation.from_quat(q,scalar_first=True).as_matrix()
        p1 = p + rot @ p1
        p2 = p + rot @ p2
        p3 = p + rot @ p3
        p4 = p + rot @ p4

        plt.cla()

        ax.plot([p1[0], p2[0], p3[0], p4[0]],
                        [p1[1], p2[1], p3[1], p4[1]],
                        [p1[2], p2[2], p3[2], p4[2]], 'k.')

        ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                    [p1[2], p2[2]], 'r-')
        ax.plot([p3[0], p4[0]], [p3[1], p4[1]],
                    [p3[2], p4[2]], 'r-')
        p_data_ = np.array(self.xs)[:,:3]
        x_data = p_data_[:, 0]
        y_data = p_data_[:, 1]
        z_data = p_data_[:, 2]
        ax.plot(x_data,y_data,z_data, 'b:')
        
        p_ref = self.x_ref[:,:3]
        ax.plot(p_ref[:,0],p_ref[:,1],p_ref[:,2],'r*')

        plt.xlim(-5, 5)
        plt.ylim(-5, 5)
        ax.set_zlim(0, 5)
        
        x_sample = self.x_mppi
        p_sample,q_sample = x_sample[:,:,0:3],x_sample[:,:,3:7]
        for p in p_sample:
            ax.plot(p[:,0],p[:,1],p[:,2],'g--')
        plt.pause(0.0001)

#现在遇到的问题，MPPI运行会特别的慢，但是计算的cost是递减的

class MPPI:
    def __init__(self, dynamics ,horizon = 20, num_samples = 20 , sigma = np.ones(4),lambda_ = 1):
        self.rk4 = Sim(QuadRotor)
        
        self.dynamic = dynamics
        self.horizon = horizon
        self.num_samples = num_samples
        self.sigma = sigma
        self.lambda_ = lambda_
        self.Q = np.eye(12)
        self.R = np.eye(4)
        self.QT = np.eye(12)
    
    def update_weights(self,Q,R,QT):
        assert Q.shape == (12,12)
        assert R.shape == (4,4)
        assert QT.shape == (12,12)
        self.Q = Q
        self.R = R
        self.QT = QT
    
    def sample(self):
        # sample trajectories
        r = np.random.normal(np.zeros(4),self.sigma,(self.num_samples,self.horizon,4))
        return r
    
    def q_error(self,q,q_des):
        #q.shape [N,H,4] q_des.shape [H,4]
        # q = [qw,qx,qy,qz] q_des = [qw,qx,qy,qz]
        # q_err = q_des*q_inv
        qd_inv_w = q_des[:,0]
        qd_inv_x = -q_des[:,1]
        qd_inv_y = -q_des[:,2]
        qd_inv_z = -q_des[:,3]
        
        q_w = q[:,:,0]
        q_x = q[:,:,1]
        q_y = q[:,:,2]
        q_z = q[:,:,3]
        
        eq_w = qd_inv_w*q_w - qd_inv_x*q_x - qd_inv_y*q_y - qd_inv_z*q_z
        eq_x = qd_inv_w*q_x + qd_inv_x*q_w + qd_inv_y*q_z - qd_inv_z*q_y
        eq_y = qd_inv_w*q_y - qd_inv_x*q_z + qd_inv_y*q_w + qd_inv_z*q_x
        eq_z = qd_inv_w*q_z + qd_inv_x*q_y - qd_inv_y*q_x + qd_inv_z*q_w
        
        e_q = np.sign(eq_w)[:,:,None]*np.concatenate([eq_x[:,:,None],eq_y[:,:,None],eq_z[:,:,None]],axis=2)
        
        return e_q
    
    def compute_cost(self,x,u,x_ref):
        # cost function
        # x.shape [N,H+1,13]
        # x_ref.shape [H,13]
        # u.shape [N,H,4]
        x_c = x[:,1:,:]
        pos_err = x_c[:,:,:3] - x_ref[:,:3]
        v_err = x_c[:,:,3:6] - x_ref[:,3:6]
        q_err = self.q_error(x_c[:,:,6:10],x_ref[:,6:10]) # vec3
        w_err = x_c[:,:,10:13] - x_ref[:,10:13]
        err = np.concatenate([pos_err,v_err,q_err,w_err],axis=2)
        cost = np.zeros((self.num_samples))
        for i in range(self.num_samples):
            for j in range(self.horizon-1):
                cost[i] += err[i,j].T @ self.Q @ err[i,j] + u[i,j].T @ self.R @ u[i,j]
            # terminal cost
            cost[i] += err[i,-1].T @ self.QT @ err[i,-1]
        return cost
    
    def mppi(self,x_0,x_ref,u_nominal):
        # x_0.shape [13,1]
        # x_ref.shape [H,13]
        # u_nominal.shape [H,4]
        
        x = np.zeros((self.num_samples,self.horizon+1,13))
        
        x[:,0,:] = x_0[:,0]
        u = np.tile(u_nominal,(self.num_samples,1,1))
        u += self.sample()
        
        # rollouts
        for i in range(self.horizon):
            # rk4 input size: x=>(13,N),u=>(4,N)
            x[:,i+1] = self.rk4.step(x[:,i].T,u[:,i].T).T
        
        # compute cost
        cost = self.compute_cost(x,u,x_ref)
        
        # compute weights
        w = np.zeros(self.num_samples)
        rou = np.min(cost)
        print("Min cost: ",rou)
        eta = np.sum(np.exp(-self.lambda_*(cost-rou)))
        w = np.exp(-self.lambda_*(cost-rou))/eta
        print("Norm weights: ",np.linalg.norm(w,1))
        u_opt = np.sum(w[:,None,None]*u,axis=0)
        np.clip(u_opt[:,0],0,20,out=u_opt[:,0])
        np.clip(u_opt[:,1:],-3,3,out=u_opt[:,1:])
        u_new = u_opt[1:] # discard the first control input
        u_new = np.vstack([u_new,u_opt[-1]])
        return u_opt[0],u_new,x

class BaselineSO3CTRL:
    def __init__(self,Kp = 1,Kd = 1, Kr = 1, Kw = 1):
        self.Kp = Kp
        self.Kd = Kd
        self.Kr = Kr
        self.Kw = Kw
            
    def vee(self,R):
        return np.array([R[2,1],R[0,2],R[1,0]])
    
    def control(self,x,x_ref):
        # x.shape [13]
        # x_ref.shape [13]
        
        pos_err = x_ref[:3] - x[:3]
        v_err = x_ref[3:6] - x[3:6] 
        
        des_acc = self.Kd*v_err + self.Kp*pos_err + np.array([0,0,QuadRotor.g])
        rot = Rotation.from_quat(x[6:10],scalar_first=True).as_matrix()
        thrust = des_acc.T @ rot @ np.array([0,0,1])
        
        yaw = 0
        e3 = des_acc/np.linalg.norm(des_acc)
        e1 = np.array([np.cos(yaw),np.sin(yaw),0])
        # y cross z = x, z cross x = y, x cross y = z
        e2 = np.cross(e3,e1)
        rot_d = np.array([e1,e2,e3]).T
        
        # rot_d = Rotation.from_quat(x_ref[6:10],scalar_first=True).as_matrix()
        
        e_rot = -self.vee(0.5*(rot_d.T @ rot - rot.T @ rot_d))
        
        w = x[10:13]
        w_d = x_ref[10:13] 
        e_omega = -(w - rot.T@rot_d@w_d)
        
        tau = self.Kr*e_rot + self.Kw*e_omega + np.cross(w,QuadRotor.J@w)

        u = np.array([thrust,tau[0],tau[1],tau[2]])
        u[0] = np.clip(u[0],0,20)
        np.clip(u[1:],-3,3,out=u[1:])
        # print(u)
        return u


def main():
    RUN_MPPI = True
    quadrotor = QuadRotor()
    mppi = MPPI(quadrotor.dynamics,num_samples=100,horizon=50,sigma=np.array([5,3,3,3]))
    vis = Visualizer()
    sim = Sim(quadrotor)
    so3_ctrl = BaselineSO3CTRL(15,5,18,2)
    mppi_Q =  np.diag([100,100,100,
            10,10,10,
            100,100,100,
            0,0,0]
        )
    mppi_R = np.diag([.1,.1,.1,.1])
    mppi_QT = mppi.num_samples*mppi_Q/13
    mppi.update_weights(mppi_Q,mppi_R,mppi_QT)

    x_0 = np.array([0,0,2,0,0,0,1,0,0,0,0,0,0]).reshape(13,1)
    t_end = 10
    dt = quadrotor.dt
    t = 0
    print("SIM STARTED")
    u_h = np.tile(np.array([quadrotor.g,0,0,0]),(mppi.horizon,1))
    x = np.zeros((1,1,13))
    while t<t_end:
        x_ref = []
        for i in range(mppi.horizon):
            # x_ref.append(Trajectory.position(t+i*dt,np.array([0,0,4])))
            x_ref.append(Trajectory.circle(t+i*dt,2,2))
        x_ref = np.array(x_ref)
        if RUN_MPPI:
                    
            u_opt, u_h, x = mppi.mppi(x_0,x_ref,u_h)
            x_0 = sim.step(x_0,u_opt[:,None])
        else:
            x = np.zeros((1,1,13))
            u = so3_ctrl.control(x_0[:,0],x_ref[0])
            x_0 = sim.step(x_0,u[:,None])
        
        vis.update(x_0,x_ref,x)
        vis.draw()
        t += dt
    
    print("SIM ENDED")
    
if __name__ == '__main__':
    main()