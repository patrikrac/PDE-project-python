# Project: Approximation of PDEs
> Developed by Patrik Rác

## Description
Implementation of a 2D finite element solver for the cours 
MU4MA029. The code is structured in to the following files:
- ```main.py``` - main file, contains the main function and the convergence benchmark.
- ```geometry.py``` - contains all methods related to the mesh generation.
- ```assembly.py``` - contains all methods related to the assembly of the local and global matrices as well as the right-hand side vector.
- ```utilites.py``` - contains other usefull methods related to visualization and the computation of the boundary elements or mesh refinement.
- ```tests.py``` - contains unit tests for the mass matrix and stiffness matrix.

The parameters concerning the funciton ```f``` are defined in the ```main()``` method of the ```main.py``` file and have to be changed in the source code if deemed nessecarry.  
By default the program sets the paramters to the values ```p=3``` and ```q=3```,  as-well as the predefined values for ```b = [1., 1.]``` and ```c=1.```. 

Additionally, a precomputed convergence plot is provided in the file ```convergence.png```, showing the quadratic convergence of the method.

## Requirements
The program is written in Python 3.10 and relies on 
libraries written for scientific computing, namely ```NumPy, SciPy and Matplotlib```.  
Furthermore, it uses the ```unittest``` library for testing.

## Usage
The program can be executed by running the following command in the terminal:  
```python3 main.py [N Nl] [--bench]```

The optional arguments are:  
- ```N``` - number of subdivisions per unit length  
- ```Nl``` - parameter degfining ```l=Nl*h``` where ```h``` is the length of the subdivision
- ```--bench``` - if present, the program will run the convergence benchmark. (This might take a while...)

By defaul (calling the program without parameters) the program will use ```N=40``` and ```Nl=20```.
It is nessecarry that ```N``` is divisible by ```Nl```, in order for the problem to be well-defined.

### Examples
```bash
python3 main.py 100 50
```
```bash
python3 main.py 100 20
```
```bash
python3 main.py --bench
```
```bash
python3 main.py 100 50 --bench
```


## Testing
Additionally, test on the mass matrix and stiffness matrix can be run by executing the following command in the terminal:   
```python3 -m unittest tests.py [-v]```  
Where the optional argument ```-v``` activates verbosity.

### Examples
```bash
python3 -m unittest tests.py
```
```bash
python3 -m unittest tests.py -v
```


