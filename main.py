"""
Created by Patrik Rác on the 19.04.2023
Main file for the project.
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse.linalg import spsolve

from geometry import *
from utilities import *
from assembly import Assemble, Assemble_optimized

from sys import argv
import time


# Define the right hand side function
def f_rhs(x, b, p=3, q=3, r=2):
    return np.exp((b[0]*x[0] + b[1]*x[1])/2.) * np.sin(p*r*np.pi*x[0]) * np.sin(q*r*np.pi*x[1])


# Define the alpha that is used in the exact solution
def alpha(b, p, q, r, c=1.):
    return 1./((b[0]**2 / 4.) + (b[1]**2 / 4.) + p**2 * r**2 * np.pi**2 + q**2 * r**2 * np.pi**2 + c)


# Function that copmutes the L2 norm of v (using the Mass Matrix M)
def L2(v, M):
    return np.sqrt(v.T@M@v)


# Error benchmark funciton that computes the relative error for different mesh sizes
def errorBenchmark(p, q, r, b=np.array([1., 1.]), c=1.):
    a = alpha(b, p, q, r, c)

    h_base = 0.1
    h_values = [h_base * 2**(-i) for i in range(6)]
    error_values = list()
    for h in h_values:
        N = int(1/h)
        print("Evaluating for N = ", N, "with a mesh size of ", h, "...")

        vtx, elt = GenerateLShapeMesh(N, int(N/r))
        print("Generated mesh with nbr of vertices: ", vtx.shape[0], " and nbr of elements: ", elt.shape[0])
        K, C, M, F = Assemble(vtx, elt, lambda x: f_rhs(x, b, p, q, r))
        A = K + C + c * M
        print("Assembled")
        U_h = spsolve(A, F)
        print("Solved")
        # Compute the exact solution
        U = np.array([a*f_rhs(x, b, p, q, r) for x in vtx])

        error_values.append(L2(U_h - U, M) / L2(U, M))

    plt.loglog(h_values, error_values, 'o-')
    plt.title("Relative error for different mesh sizes")
    plt.xlabel("h")
    plt.ylabel("Error")
    plt.gca().invert_xaxis()
    plt.grid()
    plt.show()


def timingBenchmark(p, q, r, b=np.array([1., 1.]), c=1.):
    h_base = 0.1
    h_values = [h_base * 2 ** (-i) for i in range(7)]
    for h in h_values:
        N = int(1 / h)
        print("Evaluating for N = ", N, "with a mesh size of ", h, "...")

        vtx, elt = GenerateLShapeMesh(N, int(N / r))
        print("Generated mesh with nbr of vertices: ", vtx.shape[0], " and nbr of elements: ", elt.shape[0])
        start = time.time()
        K, C, M, F = Assemble(vtx, elt, lambda x: f_rhs(x, b, p, q, r))
        A = K + C + c * M
        end = time.time()
        print("Assembly time: ", end - start, "seconds")

        start = time.time()
        U_h = spsolve(A, F)
        end = time.time()
        print("Solve time: ", end - start, "seconds")
        print()


def main():
    timingBenchmark(3, 3, 2)
    return
    # Set the default values for N and Nl
    N = 40
    Nl = 20

    # Set to True if you want to run the benchmark
    runbench = False

    # Parse the arguments
    if len(argv) > 1:
        if len(argv) == 2:
            if argv[1] == "--bench":
                runbench = True
        elif len(argv) == 3:
            N = int(argv[1])
            Nl = int(argv[2])

        elif len(argv) == 4:
            N = int(argv[1])
            Nl = int(argv[2])
            if argv[3] == "--bench":
                runbench = True
        else:
            print("Invalid number of arguments")
            exit(1)

    if N % Nl != 0:
        print("N must be a multiple of Nl")
        exit(1)

    # Generate the mesh
    vtx, elt = GenerateLShapeMesh(N, Nl)

    # Set the parameters for the function
    p = 3
    q = 3
    r = int(N / Nl)
    b = np.array([1, 1])
    c = 1.
    a = alpha(b, p, q, r, c)

    # Compute the interpolation of the exact solution on the grid
    U = np.array([a * f_rhs(x, b, p, q, r) for x in vtx])

    # Assemble the matrices and the right-hand side
    start = time.time()
    K, C, M, F = Assemble(vtx, elt, lambda x: f_rhs(x, b, p, q, r))
    A = K + C + c * M
    end = time.time()
    print("Assembly time: ", end - start, "seconds")

    start = time.time()
    U_h = spsolve(A, F)
    end = time.time()
    print("Solve time: ", end - start, "seconds")

    print("Error (L2) of the computation with N =", N, "Nl =", Nl, "is", L2(U_h - U, M) / L2(U, M))
    PlotApproximation(vtx, elt, U_h, "Approximate solution")
    PlotApproximation(vtx, elt, (U_h - U), "Error of the approximate solution")

    # Running the benchmark for the convergence plot
    if runbench:
        errorBenchmark(p, q, r, b, c)


# Main execution of the program
if __name__ == '__main__':
    main()
