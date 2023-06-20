"""
Created by Patrik Rác on the 19th of April 2023
This file contains some utility functions.
This includes mesh loading and plotting.
As well as the mesh boundary and refinement functions.
"""

import numpy as np
from matplotlib import pylab as plt


def LoadVTX(filename):
    with open(filename) as file:
        filearray = file.read().splitlines()
        nbr_vtx = int(filearray[1])
        vertecies = list()
        for i in range(2, nbr_vtx+2):
            vtx_str = filearray[i].split()
            vertecies.append(np.array([float(vtx_str[1]), float(vtx_str[2])]))

    return np.array(vertecies)


def LoadELT(filename):
    with open(filename) as file:
        el_str_list = file.read().split("$Elements")[1].splitlines()
        # Get the number of elements as the second entry in the string
        nbr_el = int(el_str_list[1])
        elem = list()
        for i in range(2, nbr_el+2):
            el_str = el_str_list[i].split()
            elem.append(np.array([int(el_str[1]), int(el_str[2]), int(el_str[3])]))

    return np.array(elem)


def PlotMesh(vtx, elt, val=None, bnd=None, nrm=None):
    """
    Plot the mesh
    :param vtx: Coordinate array
    :param elt: Connectivity array
    :param val: Value array (Specifying possible values for the color)
    :param bnd: Boundary array
    :param nrm: Normal vector array
    """
    if bnd is not None:
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        for j, edge in enumerate(bnd):
            if val is None:
                plt.plot([vtx[edge[0], 0], vtx[edge[1], 0]], [vtx[edge[0], 1], vtx[edge[1], 1]], 'k-')
            else:
                plt.plot([vtx[edge[0], 0], vtx[edge[1], 0]], [vtx[edge[0], 1], vtx[edge[1], 1]], c=colors[val[j]])

            if nrm is not None:
                plt.quiver([vtx[edge[0], 0], vtx[edge[1], 0]], [vtx[edge[0], 1], vtx[edge[1], 1]],
                           nrm[j, 0], nrm[j, 1], color='r', scale=1, scale_units='xy')
    else:
        plt.triplot(vtx[:, 0], vtx[:, 1], elt, "-")
        if val is not None:
            plt.tripcolor(vtx[:, 0], vtx[:, 1], elt, val, cmap="jet")

    plt.show()


def PlotApproximation(vtx, elt, v, title=None):
    plt.tripcolor(vtx[:, 0], vtx[:, 1], elt, v, cmap="jet")
    plt.colorbar()
    if title is not None:
        plt.title(title)
    plt.show()


def Boundary(elt):
    b = set()
    bnd_dict = dict()
    for p, tri in enumerate(elt):
        for i in range(len(tri)):
            edge = (tri[i - 1], tri[i])
            edge_rev = (tri[i], tri[i - 1])

            # Check the Hash map
            if edge not in bnd_dict:
                if edge_rev not in bnd_dict:
                    b.add(edge)
                    bnd_dict[edge] = 3 * p + ((i + 1) % 3)
                else:
                    b.remove(edge_rev)
            else:
                b.remove(edge)

    eltb = list()
    be2e = list()
    for edge in b:
        eltb.append(list(edge))
        be2e.append(bnd_dict[edge])

    return np.array(eltb), np.array(be2e)


def Refine(vtx, elt):
    # Initialize the refined mesh with the already established coarse mesh
    refined_vtx = list(vtx)
    refined_elt = list()

    n = len(vtx)

    new_vtx = dict()

    for tri in elt:
        # Compute the center of all three points
        p1 = ((vtx[tri[0]][0] + vtx[tri[1]][0]) / 2,
              (vtx[tri[0]][1] + vtx[tri[1]][1]) / 2)

        if p1 not in new_vtx:
            refined_vtx.append(np.array(p1))
            new_vtx[p1] = n
            n += 1

        p2 = ((vtx[tri[1]][0] + vtx[tri[2]][0]) / 2,
              (vtx[tri[1]][1] + vtx[tri[2]][1]) / 2)

        if p2 not in new_vtx:
            refined_vtx.append(np.array(p2))
            new_vtx[p2] = n
            n += 1

        p3 = ((vtx[tri[2]][0] + vtx[tri[0]][0]) / 2,
              (vtx[tri[2]][1] + vtx[tri[0]][1]) / 2)

        if p3 not in new_vtx:
            refined_vtx.append(np.array(p3))
            new_vtx[p3] = n
            n += 1

        refined_elt.append(np.array([tri[0], new_vtx[p1], new_vtx[p3]]))
        refined_elt.append(np.array([tri[1], new_vtx[p2], new_vtx[p1]]))
        refined_elt.append(np.array([tri[2], new_vtx[p3], new_vtx[p2]]))
        refined_elt.append(np.array([new_vtx[p1], new_vtx[p2], new_vtx[p3]]))

    return np.array(refined_vtx), np.array(refined_elt)
