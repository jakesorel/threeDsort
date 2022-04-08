import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from numba import jit
import time


class Mesh:
    def __init__(self,x,box):
        self.box = box
        self.nbox = box.shape[0]
        self.x = x

        self.X = None
        self.get_X()
        self.nc = None
        self.nC = None
        self.get_nc()

        self.delan = None
        self.tet = None
        self.neigh = None
        self.box_mask = None
        self.k2s = None

        self.tX = None
        self.v = None
        self.d2v_tX = None
        self.tX_neigh = None
        self.disp_squareform = None
        self.dist_squareform = None
        self.edges = None

        self.construct_triangulation()
        self.tetify()

    def get_nc(self):
        self.nc = self.x.shape[0]
        self.nC = self.X.shape[0]

    def get_X(self):
        self.X = get_X(self.x,self.box)

    def construct_triangulation(self):
        self.delan = Delaunay(self.X)
        self.tet = self.delan.simplices
        self.neigh= self.delan.neighbors
        self.box_mask = get_box_mask(self.tet,self.nbox)
        self.k2s = get_k2s(self.neigh,self.tet)
        # self.tet = self.tet_full[self.non_box_mask]
        # self.neigh = self.neigh_full[self.non_box_mask]

    def update_x(self,x):
        self.x = x
        self.get_X()
        self.tetify()
        is_local_delan = check_local_delaunay(self.v, self.tX_neigh, self.d2v_tX)
        if not is_local_delan:
            # print("reconstructing...")
            self.construct_triangulation()
            self.tetify()
            return True
        else:
            return False

    def tetify(self):
        self.tX = self.X[self.tet]
        self.v = get_circumcentre(self.tX)
        self.d2v_tX = get_d2_23(self.v,self.tX)
        self.tX_neigh = self.tX[self.neigh,self.k2s]
        self.tX_neigh[self.neigh==-1] = np.nan
        self.disp_squareform = get_displacements_squareform(self.tX)
        self.dist_squareform = get_L1_norm(self.disp_squareform)
        self.norm_vec_squareform = get_norm_vec(self.disp_squareform,self.dist_squareform)
        # self.v_

    def get_edges(self):
        self.edges_squareform = get_edges_squareform(self.tet)

    def get_dist_mat(self):
        self.dist_mat = get_dist_full_from_square(self.dist_squareform)

@jit(nopython=True)
def get_d2_23(x,y):
    disp = np.expand_dims(x,1) - y
    return disp[...,0]**2 + disp[...,1]**2 + disp[...,2]**2


@jit(nopython=True)
def get_box_mask(tet,nbox):
    return tet <nbox

#
# @jit(nopython=True)
# def get_non_box_mask(tet):
#     return np.sum(tet>=8, 1) == 4

@jit(nopython=True)
def get_X(x,box):
    return np.row_stack((box, x))


@jit(nopython=True)
def get_circumcentre(tx):
    a,b,c,d = tx[:,0],tx[:,1],tx[:,2],tx[:,3]
    ax,ay,az = a.T
    bx,by,bz = b.T
    cx,cy,cz = c.T
    dx,dy,dz = d.T
    denominator = 16 * (az - bz) * (
                ax * bz * cy - ax * by * cz - bz * cy * dx + by * cz * dx - ax * bz * dy + bz * cx * dy + ax * cz * dy - bx * cz * dy + az * (
                    -(bx * cy) + by * (
                        cx - dx) + cy * dx + bx * dy - cx * dy) + ax * by * dz - by * cx * dz - ax * cy * dz + bx * cy * dz + ay * (
                            -(bz * cx) + bx * cz + bz * dx - cz * dx - bx * dz + cx * dz))

    numerator_x = 4 * (2 * (ax ** 2 + ay ** 2 + az ** 2 - bx ** 2 - by ** 2 - bz ** 2) * (az - cz) - 2 * (az - bz) * (
                ax ** 2 + ay ** 2 + az ** 2 - cx ** 2 - cy ** 2 - cz ** 2)) * (
                              az * (by - dy) + bz * dy - by * dz + ay * (-bz + dz)) - 4 * (
                              az * (by - cy) + bz * cy - by * cz + ay * (-bz + cz)) * (
                              2 * (ax ** 2 + ay ** 2 + az ** 2 - bx ** 2 - by ** 2 - bz ** 2) * (az - dz) - 2 * (
                                  az - bz) * (ax ** 2 + ay ** 2 + az ** 2 - dx ** 2 - dy ** 2 - dz ** 2))

    numerator_y = -8 * (az - bz) * (
                ay ** 2 * bz * cx - ay ** 2 * bx * cz - ay ** 2 * bz * dx + bz * cx ** 2 * dx + bz * cy ** 2 * dx + ay ** 2 * cz * dx - bx ** 2 * cz * dx - by ** 2 * cz * dx - bz ** 2 * cz * dx + bz * cz ** 2 * dx - bz * cx * dx ** 2 + bx * cz * dx ** 2 - bz * cx * dy ** 2 + bx * cz * dy ** 2 + (
                    ay ** 2 * (bx - cx) + bx ** 2 * cx + (by ** 2 + bz ** 2) * cx - bx * (
                        cx ** 2 + cy ** 2 + cz ** 2)) * dz + (-(bz * cx) + bx * cz) * dz ** 2 + ax ** 2 * (
                            bz * (cx - dx) + cz * dx - cx * dz + bx * (-cz + dz)) + az ** 2 * (
                            bz * (cx - dx) + cz * dx - cx * dz + bx * (-cz + dz)) + az * (-(
                    bz ** 2 * cx) + bz ** 2 * dx - cx ** 2 * dx - cy ** 2 * dx - cz ** 2 * dx + cx * dx ** 2 + bx ** 2 * (
                                                                                                      -cx + dx) + by ** 2 * (
                                                                                                      -cx + dx) + cx * dy ** 2 + cx * dz ** 2 + bx * (
                                                                                                      cx ** 2 + cy ** 2 + cz ** 2 - dx ** 2 - dy ** 2 - dz ** 2)) + ax * (
                            (bx ** 2 + by ** 2) * (cz - dz) + bz ** 2 * (cz - dz) + (
                                cx ** 2 + cy ** 2) * dz + cz ** 2 * dz - cz * (dx ** 2 + dy ** 2 + dz ** 2) + bz * (
                                        -cx ** 2 - cy ** 2 - cz ** 2 + dx ** 2 + dy ** 2 + dz ** 2)))

    numerator_z = 8 * (az - bz) * (
                az ** 2 * by * cx - az ** 2 * bx * cy - az ** 2 * by * dx + by * cx ** 2 * dx + az ** 2 * cy * dx - bx ** 2 * cy * dx - by ** 2 * cy * dx - bz ** 2 * cy * dx + by * cy ** 2 * dx + by * cz ** 2 * dx - by * cx * dx ** 2 + bx * cy * dx ** 2 + az ** 2 * bx * dy - az ** 2 * cx * dy + bx ** 2 * cx * dy + by ** 2 * cx * dy + bz ** 2 * cx * dy - bx * cx ** 2 * dy - bx * cy ** 2 * dy - bx * cz ** 2 * dy - by * cx * dy ** 2 + bx * cy * dy ** 2 + ax ** 2 * (
                    by * (cx - dx) + cy * dx - cx * dy + bx * (-cy + dy)) + ay ** 2 * (
                            by * (cx - dx) + cy * dx - cx * dy + bx * (
                                -cy + dy)) - by * cx * dz ** 2 + bx * cy * dz ** 2 + ay * (-(
                    bz ** 2 * cx) + bz ** 2 * dx - cx ** 2 * dx - cy ** 2 * dx - cz ** 2 * dx + cx * dx ** 2 + bx ** 2 * (
                                                                                                       -cx + dx) + by ** 2 * (
                                                                                                       -cx + dx) + cx * dy ** 2 + cx * dz ** 2 + bx * (
                                                                                                       cx ** 2 + cy ** 2 + cz ** 2 - dx ** 2 - dy ** 2 - dz ** 2)) + ax * (
                            -(cy * dx ** 2) + bx ** 2 * (cy - dy) + by ** 2 * (cy - dy) + bz ** 2 * (
                                cy - dy) + cx ** 2 * dy + cy ** 2 * dy + cz ** 2 * dy - cy * dy ** 2 - cy * dz ** 2 + by * (
                                        -cx ** 2 - cy ** 2 - cz ** 2 + dx ** 2 + dy ** 2 + dz ** 2)))

    vx = numerator_x/denominator
    vy = numerator_y/denominator
    vz = numerator_z/denominator
    v = np.column_stack((vx,vy,vz))
    return v


@jit(nopython=True)
def get_k2s(neigh,tet):
    k2s = np.ones_like(tet)*-1
    for i, (tet_i,neigh_i) in enumerate(zip(tet,neigh)):
        for j in range(4):
            neigh_ij = neigh_i[j]
            if not neigh_ij == - 1:
                for k in range(4):
                    tet_neigh_i = tet[neigh_ij][k]
                    if ~np.any(tet_neigh_i == tet_i):
                        k2s[i,j] = k
    return k2s


@jit(nopython=True)
def check_local_delaunay(v,tX_neigh,d2v_tX):
    d2_v_tX_neigh = get_d2_23(v,tX_neigh)
    flip_mask = ~(d2_v_tX_neigh < d2v_tX)
    return np.all(flip_mask)

@jit(nopython=True)
def get_displacements_squareform(tX):
    """
    AB,AC,AD,BC,BD,CD
    """
    ab = tX[:,0] - tX[:,1]
    ac = tX[:,0] - tX[:,2]
    ad = tX[:,0] - tX[:,3]
    bc = tX[:,1] - tX[:,2]
    bd = tX[:,1] - tX[:,3]
    cd = tX[:,2] - tX[:,3]
    return np.dstack((ab,ac,ad,bc,bd,cd)).transpose((0,2,1))

@jit(nopython=True)
def get_edges_squareform(tX):
    """
    AB,AC,AD,BC,BD,CD
    """
    ab = np.column_stack((tX[:,0], tX[:,1]))
    ac = np.column_stack((tX[:,0], tX[:,2]))
    ad = np.column_stack((tX[:,0], tX[:,3]))
    bc = np.column_stack((tX[:,1], tX[:,2]))
    bd = np.column_stack((tX[:,1], tX[:,3]))
    cd = np.column_stack((tX[:,2], tX[:,3]))
    return np.dstack((ab,ac,ad,bc,bd,cd)).transpose((0,2,1))


@jit(nopython=True)
def get_L1_norm(X):
    return np.sqrt(X[...,0]**2 + X[...,1]**2 + X[...,2]**2)

@jit(nopython=True)
def get_norm_vec(tvec,tdist):
    return tvec/np.expand_dims(tdist,2)

@jit(nopython=True)
def get_disp_full_from_square(squareform):
    full = np.zeros((squareform.shape[0],4,4,3))
    full[:,0,1] = squareform[:,0]
    full[:,0,2] = squareform[:,1]
    full[:,0,3] = squareform[:,2]
    full[:,1,2] = squareform[:,3]
    full[:,1,3] = squareform[:,4]
    full[:,2,3] = squareform[:,5]
    full = full - full.transpose((0,2,1,3))
    return full

@jit(nopython=True)
def get_dist_full_from_square(squareform):
    full = np.zeros((squareform.shape[0],4,4))
    full[:,0,1] = squareform[:,0]
    full[:,0,2] = squareform[:,1]
    full[:,0,3] = squareform[:,2]
    full[:,1,2] = squareform[:,3]
    full[:,1,3] = squareform[:,4]
    full[:,2,3] = squareform[:,5]
    full = full + full.transpose((0,2,1))
    return full

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
#     x = np.random.uniform(0,1,(20,3))
#     mesh = Mesh(x,box)
#     mesh.construct_triangulation()
#     mesh.tetify()
#     mesh.update_x(x)
#
#     N = int(1e3)
#     t0 = time.time()
#     n_updated = 0
#     for i in range(int(1e3)):
#         x += np.random.normal(0,1e-2,x.shape)
#         out = mesh.update_x(x)
#         n_updated += int(out)
#     t1= time.time()
#     print(t1-t0)
#
#     # #
#     # t0 = time.time()
#     # for i in range(int(1e3)):
#     #     # tet_neigh = tet[mesh.neigh]
#     #     get_k2s(mesh.neigh, tet)
#     # t1= time.time()
