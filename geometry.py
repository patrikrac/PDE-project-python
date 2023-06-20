"""
Created by Patrik Rác on the 19.04.2023
Geometry file for the project
Implements the routines that are used to create the meshes
"""
import numpy as np


def GenerateRectangleMesh(Lx, Ly, Nx, Ny):
    """
    Generate a rectangular structured triangular mesh
    :param Lx: horizontal length of the domain
    :param Ly: vertical length of the domain
    :param Nx: number of horizontal divisions
    :param Ny: number of vertical divisions
    :return: (vtx, elt) tuple containing the vertex and element arrays
    """

    # Generate the vertex array
    vtx = np.array([np.array([xv, yv]) for yv in np.linspace(0, Ly, Ny+1) for xv in np.linspace(0, Lx, Nx+1)])

    # Generate the element array
    # Here we create the structured triangluation of the domain

    elt_list = list()

    for iy in range(0, Ny):
        for ix in range(0, Nx):
            elt_list.append([iy*(Nx+1) + ix, iy*(Nx+1) + ix+1, (iy+1)*(Nx+1) + ix+1])
            elt_list.append([iy*(Nx+1) + ix, (iy+1)*(Nx+1) + ix+1, (iy+1)*(Nx+1) + ix])

    return vtx, np.array(elt_list)


def GenerateLShapeMesh(N, Nl):
    """
    Generate an L-shaped structured triangular mesh
    :param N: Number of subdivisions per unit length
    :param Nl: The number of subdivions in the lower rectungular part
    :return: (vtx, elt): tuple containing the vertex and element arrays
    """

    # Compute the appropriate parameters
    h = 1/N
    l = h*Nl
    # Create two rectangular meshes according to the parameters
    # Create the lower rectangular mesh
    vtxl, eltl = GenerateRectangleMesh(1, l, N, Nl)

    if l == 1:
        return vtxl, eltl

    # Create the upper square mesh
    vtxu, eltu = GenerateRectangleMesh(l, 1-l, Nl, int((1-l)/h))

    # Shift the upper mesh upwards to fit the L-shape
    vtxu[:, 1] += l

    # Glue the two meshes together (Special attention to duplicate vertices at the interface)
    vtx = np.concatenate((vtxl, vtxu[Nl+1:]))

    # Create a dictionary with all verices
    interface_dict = {tuple(v): i for i, v in enumerate(vtxl)}

    # Loop over all elements of the upper mesh and update if nessecary
    for e in eltu:
        for i in range(0, 3):
            if tuple(vtxu[e[i]]) in interface_dict:
                e[i] = interface_dict[tuple(vtxu[e[i]])]
            else:
                e[i] += len(vtxl) - (Nl+1)

    elt = np.concatenate((eltl, eltu))

    return vtx, elt

