import numpy as np
from assimulo.explicit_ode import Explicit_ODE
from assimulo.ode import *
import assimulo.problem as apro
from wrapper import Explicit_Problem_2nd
import scipy.sparse as ssp
import scipy.sparse.linalg as ssl


class SecondOrder(Explicit_ODE):
    tol=1.e-8     
    maxit=1000     
    maxsteps=50000

    avb_methods = {'exp_newmark', 'imp_newmark', 'hht'}

    def __init__(self, problem, method, beta, gamma, alpha, C, M, K, f):
        Explicit_ODE.__init__(self, Explicit_Problem_2nd(problem))

        #Solver options
        self.options["h"] = 0.01
        
        #Statistics
        self.statistics["nsteps"] = 0
        self.statistics["nfcns"] = 0

        self.init_methods = []
        self.method = method
        self.beta = beta
        self.gamma = gamma
        self.alpha = alpha
        self.C = C
        self.M = M
        self.K = K
        self.d = self.problem.d
        self.damping = True
        self.f = f
        self.K_eff = None


        # some really messy asserts and checks that I should probably clean up at some point and prob do not work but it is something for now
        assert self.alpha >= -1/3 and self.alpha <= 0, "Alpha must be in the range [-1/3, 0]"

        if M is not None and K is not None:
            self.problem_type = "matrix"
            #if np.all(C) == 0: self.C = np.zeros_like(M)

        else:
            self.problem_type = "accel"

        if self.C is None: self.C = np.zeros_like(self.M) if self.M is not None else np.zeros((self.d, self.d))

        if self.beta == 0: # and np.all(self.C == 0):
            self.damping = False
            self.method = 'exp_newmark'
            print("No damping matrix provided (and beta = 0), switching to explicit Newmark method")

    def _set_h(self,h):
            self.options["h"] = float(h)

    def _get_h(self):
        return self.options["h"]        

    h=property(_get_h,_set_h)

    def lin_solve(self, A, b):
        if ssp.issparse(A):
            return ssl.spsolve(A, b)
        return np.linalg.solve(A, b)

    def integrate(self, t, y, tf, opts):
        h = min(self.h, abs(tf - t))

        # store history for t, u, v, a where a is given by the wrapper using rhs
        tres = [t]
        ures = [y[:self.d].copy()]
        vres = [y[self.d:].copy()]

        if self.method == 'exp_newmark':
            ares = [self.problem.accel(t, ures[-1], vres[-1])]
            
            self.statistics["nfcns"] += 1

        elif self.method == 'imp_newmark':
            ares = [self.lin_solve(self.M, self.f(tres[-1]) - self.C @ vres[-1] - self.K @ ures[-1])]
            
            #self.K_eff = self.K + self.C*self.gamma/(self.beta*h) + self.M/(self.beta*h*h) # will not work if we use variable step sizes; in that case put inside step loop (here now for computational efficiency for constant step size)
            
            self.statistics["nfcns"] += 1
        
        elif self.method == 'hht':
            self.gamma = (1/2) - self.alpha
            self.beta = (1 - self.alpha)**2 / 4

            ares = [self.lin_solve(self.M, self.f(tres[-1]) - self.C @ vres[-1] - self.K @ ures[-1])]

            self.K_eff = (
            self.M/(self.beta*h*h)
            + (1+self.alpha)*self.gamma*self.C/(self.beta*h)
            + (1+self.alpha)*self.K) 

            self.statistics["nfcns"] += 1

        for i in range(self.maxsteps):
            if tres[-1] >= tf:
                break

            self.statistics["nsteps"] += 1
            h = min(self.h, abs(tf - tres[-1]))

            t_np1, u_np1, v_np1, a_np1 = self.step(tres, ures, vres, ares, h, i)

            tres.append(t_np1)
            ures.append(u_np1.copy())
            vres.append(v_np1.copy())
            ares.append(a_np1.copy())

        else:
            raise Explicit_ODE_Exception("Final time not reached within maximum number of steps")

        return ID_PY_OK, np.array(tres), np.array(ures), np.array(vres), np.array(ares)


    def step(self, tres, ures, vres, ares, h, i):

        if self.method == 'exp_newmark':
            if i == 0: self.init_methods.append('Exp Newmark')
            return self.step_exp_newmark(tres, ures, vres, ares, h, i)

        elif self.method == 'imp_newmark':
            if i == 0: 
                self.init_methods.append('Imp Newmark')
            return self.step_imp_newmark(tres, ures, vres, ares, h, i)

        elif self.method == 'hht':
            if i == 0: 
                self.init_methods.append('HHT')
            return self.step_hht(tres, ures, vres, ares, h, i)
        
        else:
            raise Explicit_ODE_Exception("Method %s not recognized. Available methods are: %s" % (self.method, self.avb_methods))
        

    def step_exp_newmark(self, tres, ures, vres, ares, h, i):
        u, v, a = ures[-1], vres[-1], ares[-1]

        t_np1 = tres[-1] + h
        u_np1 = u + h*v + 0.5*h*h*a

        if self.problem_type == "matrix":
            # not working properly yet
            f_np1 = self.problem.accel(t_np1, u_np1, v + h*a)
            a_np1 = self.lin_solve(self.M, f_np1 - self.K @ u_np1)
        else:
            a_np1 = self.problem.accel(t_np1, u_np1, v)
        self.statistics["nfcns"] += 1

        v_np1 = v + h*((1-self.gamma)*a + self.gamma*a_np1)

        return t_np1, u_np1, v_np1, a_np1
    
    def step_imp_newmark(self, tres, ures, vres, ares, h, i):
        u, v, a = ures[-1], vres[-1], ares[-1]
        beta = self.beta
        gamma = self.gamma

        t_np1 = tres[-1] + h
        K_eff = self.K + (self.gamma/(self.beta*h))*self.C + (1/(self.beta*h*h))*self.M
        u_np1 = self.lin_solve(K_eff, self.f(t_np1) + self.M @ (ures[-1]/(beta*h*h) + vres[-1]/(beta*h) 
            + (1/(2*beta)-1)*ares[-1]) + self.C @ (gamma/(beta*h)*ures[-1] - (1 - gamma/beta)*vres[-1] - (1 - gamma/(2*beta))*ares[-1]*h))
        self.statistics["nfcns"] += 1

        v_np1 = (gamma/beta)*(u_np1 - ures[-1])/h + (1 - gamma/beta)*vres[-1] + (1- gamma/(2*beta))*ares[-1]*h
        a_np1 = (u_np1 - ures[-1])/(beta*h*h) - vres[-1]/(beta*h) - (1/(2*beta)-1)*ares[-1]

        return t_np1, u_np1, v_np1, a_np1
    
    def step_hht(self, tres, ures, vres, ares, h, i):

        u_n = ures[-1]
        v_n = vres[-1]
        a_n = ares[-1]

        beta  = self.beta
        gamma = self.gamma
        alpha = self.alpha

        t_n   = tres[-1]
        t_np1 = t_n + h

        f_n   = self.f(t_n)
        f_np1 = self.f(t_np1)

        rhs = (
            (1+alpha)*f_np1 - alpha*f_n
            + self.M @ (u_n/(beta*h*h) + v_n/(beta*h) + (1/(2*beta)-1)*a_n)
            + (1+alpha)*(self.C @ (
                gamma/(beta*h)*u_n
                - (1 - gamma/beta)*v_n
                - (1 - gamma/(2*beta))*h*a_n
            ))
            + alpha*self.C @ v_n
            + alpha*self.K @ u_n
        )

        K_eff = (1/(self.beta*h*h))*self.M + (1+self.alpha)*(self.gamma/(self.beta*h))*self.C + (1+self.alpha)*self.K
        u_np1 = self.lin_solve(K_eff, rhs)

        a_np1 = (u_np1 - u_n)/(beta*h*h) - v_n/(beta*h) - (1/(2*beta)-1)*a_n

        v_np1 = v_n + h*((1-gamma)*a_n + gamma*a_np1)

        return t_np1, u_np1, v_np1, a_np1
        

def elastic_pendulum(t, y, k: float):
    y1, y2, y3, y4 = y
    r = np.hypot(y1, y2)
    r = max(r, 1e-14)
    lam = k * (r - 1.0) / r
    return np.array([y3, y4, -y1 * lam, -y2 * lam - 1.0], dtype=float)

def initial_condition(kind: str):
    kind = kind.lower()
    if kind in ("stretched", "hi", "high"):
        return np.array([1.5, 0.0, 0.0, 1.0], dtype=float)
    if kind in ("unstretched", "lo", "low"):
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    if kind in ("stretched-fast"):
        return np.array([1.5, 0.0, 0.0, -2.0], dtype=float)
    if kind in ("unstretched-fast"):
        return np.array([1.0, 0.0, 0.0, -2.0], dtype=float)
    raise ValueError(f"Unknown IC kind: {kind}")

k = 5.0
t_end = 20.0
y0 = initial_condition("stretched")
problem = apro.Explicit_Problem(lambda t, y: elastic_pendulum(t, y, k), y0=y0)

if __name__ == '__main__':
    solver = SecondOrder(problem, method='exp_newmark', beta=0, gamma=0.5, alpha=0, C=None, M=None, K=None, f=None)
    _, t, u, v, a = solver.integrate(0.0, y0, t_end, opts={})
    import matplotlib.pyplot as plt
    plt.plot(t, u[:, 0], label='x')
    plt.plot(t, u[:, 1], label='y')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Displacement')
    plt.title('Elastic Pendulum Simulation')
    plt.show()
