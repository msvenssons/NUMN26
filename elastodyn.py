# -*- coding: utf-8 -*-
"""
@author: Peter Meisrimel, Robert Kloefkorn, Lund University
originally based on : https://comet-fenics.readthedocs.io/en/latest/demo/elastodynamics/demo_elastodynamics.py.html
"""

import os
# ensure some compilation output for this example
os.environ['DUNE_LOG_LEVEL'] = 'info'
print("Using DUNE_LOG_LEVEL=",os.getenv('DUNE_LOG_LEVEL'))

import setuptools
import matplotlib.pyplot as pl
import numpy as np

from ufl import *
from dune.ufl import Constant, DirichletBC
from dune.grid import structuredGrid as leafGridView
from dune.fem.space import lagrange as lagrangeSpace
from dune.fem.space import dgonb as dgSpace
from solver import SecondOrder

from dune.fem.operator import galerkin as galerkinOperator

import scipy.sparse as ssp
import scipy.sparse.linalg as ssl
pl.close('all')

class elastodynamic_beam:
    # Elastic parameters
    E  = 1000.0
    nu = 0.3
    mu = Constant(E / (2.0*(1.0 + nu)), name="mu")
    lmbda = Constant(E*nu / ((1.0 + nu)*(1.0 - 2.0*nu)), name="lmbda")

    # Mass density
    rho = Constant(1.0, name="rho")

    # Rayleigh damping coefficients
    eta_m = Constant(0.0, name="eta_m")
    eta_k = Constant(0.0, name="eta_k")

    def __init__(self, gridsize, T = 4.0, dimgrid=2):

        lower, upper, cells = [0., 0.], [1., 0.1], [12*gridsize, 2*gridsize]
        if dimgrid == 3:
            lower.append( 0 )
            upper.append( 0.04 )
            cells.append( gridsize )

        self.mesh = leafGridView( lower, upper, cells )
        dim = self.mesh.dimension

        # force only up to a certain time
        self.p0 = 1.
        self.cutoff_Tc = T/5
        # Define the loading as an expression
        self.p = self.p0/self.cutoff_Tc

        # Define function space for displacement (velocity and acceleration)
        self.V = lagrangeSpace( self.mesh, dimRange=dim, order=1)

        # UFL representative for x-coordinate
        x = SpatialCoordinate( self.V )

        self.bc = DirichletBC(self.V, as_vector(dim*[0]) , x[0]<1e-10)

        # Stress tensor
        def epsilon(r): # this is exactly dol.sym
            return 0.5*(nabla_grad(r) + nabla_grad(r).T)
        def sigma(r):
            return nabla_div(r)*Identity(dim) + 2.0 * self.mu * epsilon(r)

        # trial and test functions
        u = TrialFunction( self.V )
        v = TestFunction( self.V )

        # Mass form
        self.Mass_form = self.rho * inner( u, v ) * dx

        # Elastic stiffness form
        self.Stiffness_form = inner(sigma(u), epsilon(v))*dx
        # Rayleigh damping form
        self.Dampening_form = self.eta_m*self.Mass_form + self.eta_k*self.Stiffness_form

        self.M_FE = galerkinOperator([self.Mass_form,self.bc])      # mass matrix
        self.K_FE = galerkinOperator([self.Stiffness_form,self.bc]) # stiffness matrix
        self.D_FE = galerkinOperator([self.Dampening_form,self.bc]) # damping matrix

        # linear operator to assemble system matrices
        self.Mass_mat = self.M_FE.linear().as_numpy.tocsc()
        self.Stiffness_mat = self.K_FE.linear().as_numpy.tocsc()
        self.Dampening_mat = self.D_FE.linear().as_numpy.tocsc()

        # Work of external forces
        dimvec = dim*[0]
        pvec = dim*[0]
        pvec[ 1 ] = conditional(x[0] < 1 - 1e-10, 1, 0)*self.p*0.1
        pvec = as_vector(pvec)

        # right hand side
        self.External_forces_form = (inner( u, v) - inner(u,v))*dx + inner( v, pvec ) * ds

        self.F_ext = galerkinOperator(self.External_forces_form)

        self.ndofs = self.V.size
        self.F = np.zeros( self.V.size )
        self.fh = self.V.interpolate(dim*[0], name="fh")
        self.rh = self.V.interpolate(dim*[0], name="rh")

        self.F_ext( self.fh, self.rh )

        rh_np = self.rh.as_numpy
        self.F[:] = rh_np[:]

        print('degrees of freedom: ', self.ndofs)

    def res(self, t, y, yp, ypp):
        if t < self.cutoff_Tc:
            return self.Mass_mat@ypp * self.Stiffness_mat@yp + self.Dampening_mat@y - t*self.F
        else:
            return self.Mass_mat@ypp * self.Stiffness_mat@yp + self.Dampening_mat@y

    def rhs(self,t,y):
        Ft = t*self.F if t < self.cutoff_Tc else np.zeros(self.ndofs)
        return np.hstack((y[self.ndofs:],
                          ssl.spsolve(self.Mass_mat, -self.Stiffness_mat@y[:self.ndofs]
                                                     -self.Dampening_mat@y[self.ndofs:]
                                                     + Ft)))

    def evaluateAt(self, y, position):
        from dune.common import FieldVector
        from dune.fem.utility import pointSample
        from dune.generator import algorithm
        # convert vector back to DUNE function that can be sampled on the mesh
        y1 = np.array(y[:self.ndofs])
        df_y = self.V.function("df_y1", dofVector=y1 )

        if len(position) < self.mesh.dimension:
            position.append(0)

        val = pointSample( df_y, position )
        return val[ 1 ] # return displacement in y-direction

    def plotBeam( self, y ):
        # convert vector back to DUNE function
        y1 = np.array(y[:self.ndofs])
        displacement = self.V.function("displacement", dofVector=y1 )
        from dune.fem.view import geometryGridView
        x = SpatialCoordinate(self.V)
        # interpolate into coordinates for geometryGridView
        position = self.V.interpolate( x+displacement, name="position" )
        beam = geometryGridView( position )
        beam.plot()

if __name__ == '__main__':
    # test section using build-in ODE solver from Assimulo
    t_end = 10
    beam_class = elastodynamic_beam(4, T=t_end)

    def ft(t):
        return t*beam_class.F if t < beam_class.cutoff_Tc else np.zeros(beam_class.ndofs)

    import assimulo.solvers as aso
    import assimulo.ode as aode

    # y , ydot
    beam_problem = aode.Explicit_Problem(beam_class.rhs,y0=np.zeros((2*beam_class.ndofs,)))
    beam_problem.name='Modified Elastodyn example from DUNE-FEM'
    
    #beamCV = aso.ImplicitEuler(beam_problem) # CVode solver instance # implicit euler dampens the problem (eigenvalues < 1) and thus is not suitable for this problem
    """beamCV = aso.Radau5ODE(beam_problem)
    beamCV.h = 0.05 # constant step size here
    tt, y = beamCV.simulate(t_end)

    disp_tip = []
    plottime = 0
    plotstep = 0.25
    for i, t in enumerate(tt):
        disp_tip.append(beam_class.evaluateAt(y[i], [1, 0.05]))
        if t > plottime:
            print(f"Beam position at t={t}")
            #beam_class.plotBeam( y[i] )
            plottime += plotstep

    pl.figure()
    pl.plot(tt, disp_tip, '-b')
    pl.title('Displacement of beam tip over time')
    pl.xlabel('t')
    pl.ylabel('Displacement')
    pl.savefig('displacement.png', dpi = 200)
    pl.show()"""


    # solve using our Newmark-beta/HHT method
    solver = SecondOrder(
    beam_problem,
    method="imp_newmark",
    beta=0.4,
    gamma=0.5,
    alpha=-0.2,
    C=beam_class.Dampening_mat,
    M=beam_class.Mass_mat,
    K=beam_class.Stiffness_mat,
    f=ft,
    )
    solver.h = 0.05

    _, t, u, v, a = solver.integrate(0.0, beam_problem.y0, t_end, opts={})

    disp_tip = []
    for i, ti in enumerate(t):
        y_full = np.hstack((u[i], v[i]))
        disp_tip.append(beam_class.evaluateAt(y_full, [1, 0.05]))

    pl.figure()
    pl.plot(t, disp_tip)
    pl.title("Beam tip displacement")
    pl.xlabel("t")
    pl.show()

