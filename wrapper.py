import numpy as np
import assimulo.problem as apro

class Explicit_Problem_2nd(apro.Explicit_Problem):
    def __init__(self, first_order_problem):
        self.problem = first_order_problem

        y0 = np.asarray(first_order_problem.y0, dtype=float)
        self.d = len(y0) // 2
        self.u0 = y0[:self.d].copy()
        self.v0 = y0[self.d:].copy()
        super().__init__(rhs=self.rhs, y0=y0)

        self.name = "Second order explicit problem"

    def rhs(self, t, y):
        return self.problem.rhs(t, y)

    def accel(self, t, u, v):
        y = np.hstack((u, v))
        ydot = self.problem.rhs(t, y)
        return ydot[self.d:]



if __name__ == '__main__':
    from elastodyn import elastodynamic_beam
    t_end = 8
    beam_class = elastodynamic_beam(4, T=t_end)
    problem = Explicit_Problem_2nd(apro.Explicit_Problem(beam_class.rhs,y0=np.zeros((2*beam_class.ndofs,))))

