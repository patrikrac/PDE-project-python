#
# Created by Patrik Rác
# Test script for the EDP project.
# Mainly test the assembled Mass and Stiffness matrices.
#

import unittest as ut

from geometry import *
from assembly import *
from utilities import *


class TestMass(ut.TestCase):
    def test_mass(self):
        """
        Testing the mass matrix condition U.t*M*U = A
        """
        N = 10
        vtx, elt = GenerateLShapeMesh(N, int(N/2))
        M = Mass(vtx, elt, bnd_vtx=set())
        U = np.ones(vtx.shape[0])
        self.assertAlmostEqual(U.T@M@U, 0.75, delta=1e-10)


class TestStiffness(ut.TestCase):
    def test_stiffness(self):
        """
        Testing the stiffness matrix condition K*U = 0
        """
        N = 10
        vtx, elt = GenerateLShapeMesh(N, int(N/2))
        K = Rig(vtx, elt, bnd_vtx=set())
        U = np.ones(vtx.shape[0])
        self.assertAlmostEqual(np.linalg.norm(K@U), 0, delta=1e-10)

    def test_stiffness_2(self):
        """
        Testing the stiffness matrix condition V.T*K*U = alpha_v*alpha_u*A
        """
        def u(x, alpha, beta): return np.dot(alpha, x) + beta
        N = 10
        alpha_v = np.random.rand(2)
        beta_v = np.random.rand()
        alpha_u = np.random.rand(2)
        beta_u = np.random.rand()

        vtx, elt = GenerateLShapeMesh(N, int(N/2))
        K = Rig(vtx, elt, bnd_vtx=set())

        U = np.array([u(x, alpha_u, beta_u) for x in vtx])
        V = np.array([u(x, alpha_v, beta_v) for x in vtx])

        self.assertAlmostEqual(V.T@K@U, np.dot(alpha_v,alpha_u)*0.75, delta=1e-10)


if __name__ == '__main__':
    ut.main()
