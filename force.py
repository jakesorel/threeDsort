import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from numba import jit
import time
from mesh import Mesh
import tet_functions as tef


class Force:
    def __init__(self,tissue):
        self.t = tissue
        self.F = None
        self.get_force()

    def get_force(self):
        self.F = get_forces(self.t.mesh.norm_vec_squareform,self.t.mesh.dist_squareform,self.t.tissue_params["dmax"],self.t.kappa_adhesion_squareform,self.t.kappa_repulsion_squareform,self.t.mesh.tet)



class ActiveForce:
    """
    Active force class
    ------------------

    Calculates the active forces acting on a cell centroid. This is traditionally phrased in terms of v0 and Dr, being the fixed velocity and the rotational diffusion of the direction.
    """

    def __init__(self, tissue, active_params=None):
        assert active_params is not None, "Specify active params"
        self.t = tissue
        self.active_params = active_params
        self.aF = None
        self.theta = np.random.uniform(0, np.pi * 2, self.t.mesh.nc)
        self.phi = np.random.uniform(0, np.pi * 2, self.t.mesh.nc)
        self.get_active_force()
        if type(self.active_params["v0"]) is float:
            self.active_params["v0"] = self.active_params["v0"] * np.ones(self.t.mesh.nc)

    def update_active_param(self, param_name, val):
        self.active_params[param_name] = val

    def update_orientation(self, dt):
        """
        Time-steps the orientation (angle of velocity) according to the equation outlined in Bi et al PRX.
        :param dt:
        :return:
        """
        # self.orientation = _update_persistent_random_orientation(self.orientation,
        #                                                    self.active_params["Dr"],
        #                                                    dt,
        #                                                    self.t.mesh.n_c)
        self.theta = _update_persistent_random_orientation(self.theta,
                                                           self.active_params["Dr"],
                                                           dt,
                                                           self.t.mesh.nc)
        self.phi = _update_persistent_random_orientation(self.phi,
                                                           self.active_params["Dr"],
                                                           dt,
                                                           self.t.mesh.nc)


    @property
    def orientation_vector(self):
        """
        Property. Converts angle to a unit vector
        :return: Unit vector
        """
        return _vec_from_angle(self.theta,self.phi)

    def get_active_force(self):
        """
        Standard SPV model
        :return:
        """
        self.aF = _get_active_force(self.orientation_vector,
                                    self.active_params["v0"])

    def update_active_force(self, dt):
        self.update_orientation(dt)
        self.get_active_force()
        return self.aF

    ##but could include other options here...


@jit(nopython=True)
def _get_active_force(orientation, v0):
    return (v0 * orientation.T).T


@jit(nopython=True)
def _update_persistent_random_orientation(orientation, Dr, dt, n_c):
    return (orientation + np.random.normal(0, np.sqrt(2 * Dr * dt), n_c))%(np.pi*2)



@jit(nopython=True)
def _vec_from_angle(theta,phi):
    return np.column_stack((np.cos(theta)*np.sin(phi), np.cos(theta)*np.cos(phi),np.sin(theta)))



@jit(nopython=True)
def get_force_components(norm_vec_squareform,dist_squareform,dmax,kappa_adhesion_squareform,kappa_repulsion_squareform):
    force_mask = dist_squareform <= dmax
    adhesion_force_mag = kappa_adhesion_squareform*dist_squareform
    repulsion_force_mag = kappa_repulsion_squareform*(dmax-dist_squareform)
    force_mag = adhesion_force_mag - repulsion_force_mag
    force_mag = force_mag*(force_mask*1.0)
    force_components = np.expand_dims(force_mag,2)*norm_vec_squareform
    return force_components

@jit(nopython=True)
def get_tforce(nv,force_components):
    tforce = np.zeros((nv,4,3))
    tforce[:,0] -= force_components[:,0]
    tforce[:,0] -= force_components[:,1]
    tforce[:,0] -= force_components[:,2]
    tforce[:,1] -= force_components[:,3]
    tforce[:,1] -= force_components[:,4]
    tforce[:,2] -= force_components[:,5]

    tforce[:,1] += force_components[:,0]
    tforce[:,2] += force_components[:,1]
    tforce[:,3] += force_components[:,2]
    tforce[:,2] += force_components[:,3]
    tforce[:,3] += force_components[:,4]
    tforce[:,3] += force_components[:,5]
    return tforce

def get_forces(norm_vec_squareform,dist_squareform,dmax,kappa_adhesion_squareform,kappa_repulsion_squareform,tet):
    nv = norm_vec_squareform.shape[0]
    nc = np.max(tet) + 1
    force_components = get_force_components(norm_vec_squareform, dist_squareform, dmax, kappa_adhesion_squareform,
                         kappa_repulsion_squareform)
    tforce = get_tforce(nv, force_components)
    force = tef.assemble_tet3(tforce,tet,nc)
    return force


#
#
#
# if __name__ == "__main__":
#     box = np.array(((-1,-1,-1),
#                     (-1,-1,2),
#                     (-1,2,-1),
#                     (2,-1,-1),
#                     (-1,2,2),
#                     (2,-1,2),
#                     (2,2,-1),
#                     (2,2,2)))*100.0
#     box += np.random.normal(0,1e-5,box.shape)
#     x = np.random.uniform(0,2,(60,3))
#     mesh = Mesh(x,box)
#     tissue = Tissue(mesh)
#     force = Force(tissue)
#     self = force
#
#     dmax = 1
#     kappa_adhesion_squareform = np.ones_like(self.t.mesh.dist_squareform)*1.0
#     kappa_repulsion_squareform = np.ones_like(self.t.mesh.dist_squareform)*1.0
#
#     fig, ax = plt.subplots()
#     d = np.linspace(0,dmax)
#     ax.plot(1*d - (dmax-d)*1)
#     fig.show()
#
#
#     t0 = time.time()
#     dt = 0.01
#     tfin = 10
#     t_span = np.arange(0,tfin,dt)
#     nt = t_span.size
#     x_save = np.zeros(((nt,)+x.shape))
#     for i,t in enumerate(t_span):
#         kappa_adhesion_squareform = np.ones_like(self.t.mesh.dist_squareform) * 2.0
#         kappa_repulsion_squareform = np.ones_like(self.t.mesh.dist_squareform) * 1.0
#         frc = get_forces(self.t.mesh.norm_vec_squareform,self.t.mesh.dist_squareform,dmax,kappa_adhesion_squareform,kappa_repulsion_squareform,self.t.mesh.tet)
#         x += dt*frc[8:]
#         x += np.random.uniform(0,1e-2,x.shape)
#         x_save[i] = x.copy()
#         self.t.mesh.update_x(x)
#     t1= time.time()
#     print(t1-t0)
#
#
# """
# To do next:
#
# Get the differential adhesion working
# Sort the active propulsion to work properly
#
# Adjust the 'box' towards a pyramid, to map to their set up.
#
# Get a better initialisation?
#
# Think about whether this is a good description.
#
# Think about 'contractility':
# - contractility will be summed over all of the possible neighbours deforming a cell
# - ii
# """
#
#
# x = x_save[-1]
#
# import plotly.io as pio
# pio.renderers.default = "browser"
# import plotly.express as px
# df = px.data.iris()
# fig = px.scatter_3d(x = x[:,0],y=x[:,1],z=x[:,2])
# fig.show()