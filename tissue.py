import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from numba import jit
import time
from mesh import Mesh
from force import Force

class Tissue:
    def __init__(self,mesh=None,tissue_params=None,init_params=None):
        self.tissue_params = tissue_params
        self.init_params = init_params

        if mesh is None:
            self.mesh = None
            self.initialize_x()
        else:
            self.mesh = mesh

        self.c_types,self.c_types_real = None,None

        self.kappa_adhesion_mat = None
        self.kappa_repulsion_mat = None
        self.kappa_adhesion_squareform = None
        self.kappa_repulsion_squareform = None

        self.assign_c_types()
        self.get_kappa_matrix()
        self.get_kappa_squareform()
        self.force = Force(self)

    def initialize_x(self):
        nc = self.init_params["n_E"] + self.init_params["n_T"] + self.init_params["n_X"]
        x = np.random.uniform(self.init_params["xlim"][0],self.init_params["xlim"][1],(nc,3))
        self.mesh = Mesh(x,self.init_params["box"])

    def assign_c_types(self):
        self.c_types_real = np.zeros(self.mesh.nc,dtype=int)
        self.c_types_real[:self.init_params["n_E"]] = 0
        self.c_types_real[self.init_params["n_E"]:self.init_params["n_E"]+self.init_params["n_T"]] = 1
        self.c_types_real[self.init_params["n_E"]+self.init_params["n_T"]:] = 2
        self.c_types = np.ones(self.mesh.nC,dtype=int)*3
        self.c_types[self.mesh.nbox:] = self.c_types_real


    def get_kappa_matrix(self):
        self.kappa_adhesion_mat = np.zeros((self.mesh.nC,self.mesh.nC))
        for i in range(4):
            for j in range(4):
                self.kappa_adhesion_mat += (self.c_types == i)*np.expand_dims(self.c_types == j,1)*self.tissue_params["kappa_adhesion"][i,j]
        self.kappa_adhesion_mat *= (1-np.eye(self.mesh.nC))

        self.kappa_repulsion_mat = np.zeros((self.mesh.nC, self.mesh.nC))
        for i in range(4):
            for j in range(4):
                self.kappa_repulsion_mat += (self.c_types == i) * np.expand_dims(self.c_types == j, 1) * \
                                           self.tissue_params["kappa_repulsion"][i, j]
        self.kappa_repulsion_mat *= (1 - np.eye(self.mesh.nC))

    def get_kappa_squareform(self):
        self.kappa_adhesion_squareform = get_kappa_squareform(self.kappa_adhesion_mat,self.mesh.tet)
        self.kappa_repulsion_squareform = get_kappa_squareform(self.kappa_repulsion_mat,self.mesh.tet)

    def update_x(self,x):
        retriangulated = self.mesh.update_x(x)
        if retriangulated:
            self.get_kappa_squareform()
        self.force.get_force()

@jit(nopython=True)
def get_kappa_squareform(kappa_mat,tet):
    kappa_squareform = np.zeros((tet.shape[0],6))
    for i, tt in enumerate(tet):
        kappa_squareform[i,0] = kappa_mat[tt[0],tt[1]]
        kappa_squareform[i,1] = kappa_mat[tt[0],tt[2]]
        kappa_squareform[i,2] = kappa_mat[tt[0],tt[3]]
        kappa_squareform[i,3] = kappa_mat[tt[1],tt[2]]
        kappa_squareform[i,4] = kappa_mat[tt[1],tt[3]]
        kappa_squareform[i,5] = kappa_mat[tt[2],tt[3]]
    return kappa_squareform



