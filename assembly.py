"""
Created by Patrik Rác on the 19.04.2023
Assembly file that contains code for the assembly of the matrices and the right-hand side
"""

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from utilities import Boundary

import time


def Mloc(vtx, e):
    """
    Compute the local mass matrix
    :param vtx: Vertex array
    :param e: Current element
    :return: M_loc: Local mass matrix
    """

    n = len(e)

    if n == 2:
        # Get the vertices of the current element
        v1 = vtx[e[0]]
        v2 = vtx[e[1]]

        # Compute the length of the edge
        L = np.sqrt((v1[0] - v2[0]) ** 2 + (v1[1] - v2[1]) ** 2)

        # Compute the local mass matrix
        M = L / 6. * np.array([[2, 1],
                               [1, 2]])
    elif n == 3:
        # Get the vertices of the current element
        v1 = vtx[e[0]]
        v2 = vtx[e[1]]
        v3 = vtx[e[2]]

        # Compute the area of the triangle
        A = np.abs((v1[0] - v3[0]) * (v2[1] - v1[1]) - (v1[0] - v2[0]) * (v3[1] - v1[1])) / 2.

        # Compute the local mass matrix
        M = A / 12. * np.array([[2, 1, 1],
                                [1, 2, 1],
                                [1, 1, 2]])

    else:
        print("Error: element not supported")
        return None

    return M


def Mass(vtx, elt, bnd_vtx=None) -> coo_matrix:
    """
    Compute the global mass matrix
    :param vtx: Vertex array
    :param elt: Element array
    :param bnd_vtx: Set of boundary vertices
    :return: coo_matrix: Global mass matrix
    """
    if bnd_vtx is None:
        bnd_vtx = set()

    # Compute the global mass matrix
    nbr_vtx = len(vtx)
    nbr_elt = len(elt)
    d = len(elt[0])

    V = np.zeros((d, d, nbr_elt))
    I = np.zeros((d, d, nbr_elt))
    J = np.zeros((d, d, nbr_elt))
    for i, e in enumerate(elt):
        # Compute the local mass matrix
        M_loc = Mloc(vtx, e)
        # Add the local mass matrix to the global mass matrix
        for j in range(d):
            for k in range(d):
                # Check if the current vertex is not on the boundary
                if (e[j] not in bnd_vtx) and (e[k] not in bnd_vtx):
                    I[j, k, i] = e[j]
                    J[j, k, i] = e[k]
                    V[j, k, i] = M_loc[j, k]
                elif e[j] == e[k]:
                    I[j, k, i] = e[j]
                    J[j, k, i] = e[k]
                    V[j, k, i] = 1.

    return coo_matrix((V.flat, (I.flat, J.flat)), shape=(nbr_vtx, nbr_vtx))


def Kloc(vtx, e):
    """
    Compute the local stiffness matrix
    :param vtx: Vertex array
    :param e: Current element
    :return: Local stiffness matrix
    """

    # Get the vertices of the current element
    v1 = vtx[e[0]]
    v2 = vtx[e[1]]
    v3 = vtx[e[2]]

    # Compute the area of the triangle
    A = np.abs((v1[0] - v3[0]) * (v2[1] - v1[1]) - (v1[0] - v2[0]) * (v3[1] - v1[1])) / 2.

    # Compute the local stiffness matrix
    K = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            K[i, j] = np.dot(vtx[e[(i + 1) % 3]] - vtx[e[(i + 2) % 3]], vtx[e[(j + 1) % 3]] - vtx[e[(j + 2) % 3]])

    return K / (4. * A)


def Rig(vtx, elt, bnd_vtx=None) -> coo_matrix:
    """
    Compute the global stiffness matrix
    :param vtx: Vertex array
    :param elt: Element array
    :param bnd_vtx: Set of boundary vertices
    :return: Global stiffness matrix
    """

    if bnd_vtx is None:
        bnd_vtx = set()

    # Compute the global stiffness matrix
    nbr_vtx = len(vtx)
    nbr_elt = len(elt)
    d = len(elt[0])
    V = np.zeros((d, d, nbr_elt))
    I = np.zeros((d, d, nbr_elt))
    J = np.zeros((d, d, nbr_elt))
    for i, e in enumerate(elt):
        K_loc = Kloc(vtx, e)
        for j in range(d):
            for k in range(d):
                # Check if the current vertex is not on the boundary
                if (e[j] not in bnd_vtx) and (e[k] not in bnd_vtx):
                    I[j, k, i] = e[j]
                    J[j, k, i] = e[k]
                    V[j, k, i] = K_loc[j, k]
                elif e[j] == e[k]:
                    I[j, k, i] = e[j]
                    J[j, k, i] = e[k]
                    V[j, k, i] = 1.

    return coo_matrix((V.flat, (I.flat, J.flat)), shape=(nbr_vtx, nbr_vtx))


def Cloc(vtx, e, b=np.array([1, 1])):
    """
    Compute the local center matrix
    :param vtx: Vertex array
    :param e: Current element
    :param b: Vector b
    :return: Local center matrix
    """

    # Get the vertices of the current element
    v1 = vtx[e[0]]
    v2 = vtx[e[1]]
    v3 = vtx[e[2]]

    # Compute the area of the triangle
    A = np.abs((v1[0] - v3[0]) * (v2[1] - v1[1]) - (v1[0] - v2[0]) * (v3[1] - v1[1])) / 2.

    # Compute the local stiffness matrix
    C = np.zeros((3, 3))
    for k in range(3):
        # Compute the associate normal vector
        nk = np.cross(np.array([0, 0, 1]), vtx[e[(k + 1) % 3]] - vtx[e[(k + 2) % 3]])[:2]
        nk = nk / np.linalg.norm(nk)
        C[:, k] = np.dot(b, nk) / np.dot((vtx[e[k]] - vtx[e[(k + 1) % 3]]), nk)

    return A / 3. * C


def Cmat(vtx, elt, bnd_vtx=None, b=np.array([1, 1])) -> coo_matrix:
    """
    Compute the global center matrix
    :param vtx: Vertex array
    :param elt: Element array
    :param bnd_vtx: Set of boundary vertices
    :param b: Vector b
    :return: Global center matrix
    """

    if bnd_vtx is None:
        bnd_vtx = set()

    # Compute the global divergence matrix
    nbr_vtx = len(vtx)
    nbr_elt = len(elt)
    d = len(elt[0])

    V = np.zeros((d, d, nbr_elt))
    I = np.zeros((d, d, nbr_elt))
    J = np.zeros((d, d, nbr_elt))
    for i, e in enumerate(elt):
        C_loc = Cloc(vtx, e, b)
        for j in range(d):
            for k in range(d):
                # Check if the current vertex is on the boundary
                if (e[j] not in bnd_vtx) and (e[k] not in bnd_vtx):
                    I[j, k, i] = e[j]
                    J[j, k, i] = e[k]
                    V[j, k, i] = C_loc[j, k]
                elif e[j] == e[k]:
                    I[j, k, i] = e[j]
                    J[j, k, i] = e[k]
                    V[j, k, i] = 1.

    return coo_matrix((V.flat, (I.flat, J.flat)), shape=(nbr_vtx, nbr_vtx))


def Floc(vtx, e, f):
    # Get the vertices of the current element
    v1 = vtx[e[0]]
    v2 = vtx[e[1]]
    v3 = vtx[e[2]]

    # Compute the area of the triangle
    A = np.abs((v1[0] - v3[0]) * (v2[1] - v1[1]) - (v1[0] - v2[0]) * (v3[1] - v1[1])) / 2.
    F_loc = np.zeros(3)
    for i in range(3):
        ni = np.cross(np.array([0, 0, 1]), vtx[e[(i + 1) % 3]] - vtx[e[(i + 2) % 3]])[:2]
        def g(x): return f(x) * np.dot(x - vtx[e[(i + 1) % 3]], ni) / np.dot(vtx[e[i]] - vtx[e[(i + 1) % 3]], ni)
        F_loc[i] = g((v1 + v2) / 2.) + g((v2 + v3) / 2.) + g((v3 + v1) / 2.)

    # Compute the local stiffness matrix
    return A / 3. * F_loc


def assemble_rhs(vtx, elt, f, bnd_vtx=None):
    """
    Assemble the right hand side
    :param vtx: Vertex array
    :param elt: Element array
    :param f:  Right hand side function f
    :param bnd_vtx: Set of boundary vertices
    :return: Right hand side vector F
    """
    if bnd_vtx is None:
        bnd_vtx = set()

    F = np.zeros(len(vtx))
    for e in elt:
        F_loc = Floc(vtx, e, f)
        for i in range(3):
            if e[i] not in bnd_vtx:
                F[e[i]] += F_loc[i]

    return F


def Assemble(vtx, elt, f, b=np.array([1, 1]), c=1.):
    """
    Assemble the global matrix A (in parts of K, C, and M) and the right hand side F
    :param vtx: Vertex arrray
    :param elt: Element array
    :param f: Right hand side function f
    :param b: Vector b
    :param c: Parameter c
    :return: Global matrices K, C and M and the right hand side vector F
    """

    # Prepare treatment of the boundary values
    bnd, _ = Boundary(elt)
    # Create a set that stores all the boundary vertices
    bnd_vtx = {v for e in bnd for v in e}

    # Assemble the global matrices
    M = Mass(vtx, elt, bnd_vtx)
    K = Rig(vtx, elt, bnd_vtx)
    C = Cmat(vtx, elt, bnd_vtx, b)

    # A = K + C + c * M

    # Assemble the right hand side
    F = assemble_rhs(vtx, elt, f, bnd_vtx)

    return K, C, M, F.T


def Assemble_optimized(vtx, elt, f, b=np.array([1, 1]), c=1.):
    """
    Directly Assemble the global matrix A (if K, C and M are not needed) and the right hand side F
    :param vtx: Vertex arrray
    :param elt: Element array
    :param f: Right hand side function f
    :param b: Vector b
    :param c: Parameter c
    :return: Global matrix A and the right hand side vector F
    """

    # Prepare treatment of the boundary values
    bnd, _ = Boundary(elt)
    # Create a set that stores all the boundary vertices
    bnd_vtx = {v for e in bnd for v in e}

    # Compute the global matrix
    nbr_vtx = len(vtx)
    nbr_elt = len(elt)
    d = len(elt[0])

    V = np.zeros((d, d, nbr_elt))
    I = np.zeros((d, d, nbr_elt))
    J = np.zeros((d, d, nbr_elt))
    for i, e in enumerate(elt):
        M_loc = Mloc(vtx, e)
        K_loc = Kloc(vtx, e)
        C_loc = Cloc(vtx, e, b)
        for j in range(d):
            for k in range(d):
                # Check if the current vertex is on the boundary
                if (e[j] not in bnd_vtx) and (e[k] not in bnd_vtx):
                    I[j, k, i] = e[j]
                    J[j, k, i] = e[k]
                    V[j, k, i] = K_loc[j, k] + C_loc[j, k] + c * M_loc[j, k]
                elif e[j] == e[k]:
                    I[j, k, i] = e[j]
                    J[j, k, i] = e[k]
                    V[j, k, i] = 1.

    A = coo_matrix((V.flat, (I.flat, J.flat)), shape=(nbr_vtx, nbr_vtx))
    A.sum_duplicates()

    # Assemble the right hand side
    F = assemble_rhs(vtx, elt, f)

    return A, F.T
