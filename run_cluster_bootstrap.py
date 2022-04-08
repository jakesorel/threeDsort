import numpy as np
from simulation import Simulation
from mesh import Mesh
import sys
import os
from numba import jit
import pandas as pd
from scipy.stats import nakagami
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import nakagami

if __name__ is "__main__":

    if not os.path.exists("results"):
        os.mkdir("results")

    if not os.path.exists("results/raw"):
        os.mkdir("results/raw")

    if not os.path.exists("results/processed"):
        os.mkdir("results/processed")

    iter_i,rep_j = int(sys.argv[1]),int(sys.argv[2])

    box = np.array(((-1,-1,-1),
                    (-1,-1,2),
                    (-1,2,-1),
                    (2,-1,-1),
                    (-1,2,2),
                    (2,-1,2),
                    (2,2,-1),
                    (2,2,2)))*100.0


    d0 = 0.7
    dmax = 1

    def get_rep(kA,dmax,d0):
        return d0*kA/(dmax-d0)

    tissue_params = {"kappa_adhesion":np.array(((0,0,0,0),(0,0,0,0),(0,0,0,0),(0,0,0,0))),
                     "kappa_repulsion":np.array(((0,0,0,1),(0,0,0,1),(0,0,0,1),(1,1,1,1))),
                     "dmax":1.0}
    init_params = {"n_E":12,"n_T":20,"n_X":10,"box":box,"xlim":[-1,1]}

    active_params = {"v0":2,"Dr":10}
    sim_params = {"dt":0.01,"tfin":200,"skip":50}

    sim = Simulation(tissue_params,init_params,active_params,sim_params)
    adhesion_vals_full = np.load("adhesion_matrices/%i.npz"%iter_i).get("adhesion_vals")
    sim.t.kappa_adhesion_mat = adhesion_vals_full
    sim.t.kappa_repulsion_mat =  get_rep(adhesion_vals_full,dmax,d0)
    sim.simulate()
    sim.save_skeleton("%d_%d"%(iter_i,rep_j),"results")

    ###ANALYSIS


    connected_components_save = np.zeros((sim.nt_save,3),dtype=int)
    for i, x in enumerate(sim.x_save):
        mesh = Mesh(x,init_params["box"])
        tet = mesh.tet
        mesh.get_edges()
        edges = mesh.edges_squareform[mesh.dist_squareform<=tissue_params["dmax"]]
        adj = coo_matrix(([True]*len(edges),(edges[:,0],edges[:,1])),shape=(mesh.nC,mesh.nC))
        adj += adj.T

        c_type = 0
        connected_components_save[i,0] = connected_components(adj[(sim.t.c_types==c_type)].T[(sim.t.c_types==c_type)].T)[0]
        c_type = 1
        connected_components_save[i,1] = connected_components(adj[(sim.t.c_types==c_type)].T[(sim.t.c_types==c_type)].T)[0]
        c_type = 2
        connected_components_save[i, 2] = connected_components(adj[(sim.t.c_types == c_type)].T[(sim.t.c_types == c_type)].T)[0]

    df_cc = pd.DataFrame({"t":sim.t_span_save,
                  "ES_connected":connected_components_save[:,0],
                  "TS_connected":connected_components_save[:,1],
                  "XEN_connected":connected_components_save[:,2]})

    if not os.path.exists("results/processed/connected_components"):
        os.mkdir("results/processed/connected_components")

    df_cc.to_csv("results/processed/connected_components/%d_%d.csv"%(iter_i,rep_j),index=None)




    E_types = np.expand_dims(sim.t.c_types_real == 0,1)
    nE = E_types.sum()

    T_types = np.expand_dims(sim.t.c_types_real == 1,1)
    nT = T_types.sum()

    @jit(nopython=True)
    def sample_random_d_centroid(x,c_types_real_):
        c_types_real = c_types_real_.copy()
        np.random.shuffle(c_types_real)
        E_types = np.expand_dims(c_types_real == 0, 1)
        nE = E_types.sum()
        T_types = np.expand_dims(c_types_real == 1, 1)
        nT = T_types.sum()
        E_centroid = (x*E_types).sum(axis=0)/nE
        T_centroid = (x*T_types).sum(axis=0)/nT
        dcentroid = np.sqrt(((E_centroid-T_centroid)**2).sum())
        return dcentroid


    dcentroid = np.zeros(len(sim.x_save))
    bias = np.zeros(len(sim.x_save))
    bias_p = np.zeros(len(sim.x_save))

    for i, x in enumerate(sim.x_save):
        E_centroid = (x*E_types).sum(axis=0)/nE
        T_centroid = (x*T_types).sum(axis=0)/nT
        dcentroid[i] = np.linalg.norm(E_centroid-T_centroid)
        d_centroid_sample = np.array([sample_random_d_centroid(x, sim.t.c_types_real) for i in range(int(1e3))])
        d_centroid_i = np.linalg.norm(E_centroid-T_centroid)
        bias[i] = np.linalg.norm(E_centroid-T_centroid)/d_centroid_sample.mean()
        bias_p[i] = 1-nakagami.cdf(d_centroid_i,*nakagami.fit(d_centroid_sample))

    df_et_disp = pd.DataFrame({"t":sim.t_span_save,"dcentroid":dcentroid,"bias":bias,"bias_p":bias_p})
    if not os.path.exists("results/processed/ET_displacement"):
        os.mkdir("results/processed/ET_displacement")


    df_et_disp.to_csv("results/processed/ET_displacement/%d_%d.csv"%(iter_i,rep_j),index=None)


    es_ids = np.nonzero(sim.t.c_types == 0)[0]
    ts_ids = np.nonzero(sim.t.c_types == 1)[0]
    xen_ids = np.nonzero(sim.t.c_types == 2)[0]
    frac_outside = np.zeros((sim.x_save.shape[0],3))
    for i, x in enumerate(sim.x_save):
        mesh = Mesh(x,init_params["box"])
        outsides = np.zeros_like(es_ids)
        for j, xen_id in enumerate(es_ids):
            xen_tets = mesh.tet[np.nonzero(np.sum(mesh.tet == xen_id,axis=1))[0]]
            outsides[j] = (xen_tets<8).any()
        frac_outside[i,0] = (outsides!=0).mean()
        outsides = np.zeros_like(ts_ids)
        for j, xen_id in enumerate(ts_ids):
            xen_tets = mesh.tet[np.nonzero(np.sum(mesh.tet == xen_id,axis=1))[0]]
            outsides[j] = (xen_tets<8).any()
        frac_outside[i,1] = (outsides!=0).mean()
        outsides = np.zeros_like(xen_ids)
        for j, xen_id in enumerate(xen_ids):
            xen_tets = mesh.tet[np.nonzero(np.sum(mesh.tet == xen_id,axis=1))[0]]
            outsides[j] = (xen_tets<8).any()
        frac_outside[i,2] = (outsides!=0).mean()

    df_outside = pd.DataFrame({"t":sim.t_span_save,
                               "ES_frac_outside":frac_outside[:,0],
                               "TS_frac_outside":frac_outside[:,1],
                               "XEN_frac_outside":frac_outside[:,2]})

    if not os.path.exists("results/processed/outside"):
        os.mkdir("results/processed/outside")


    df_outside.to_csv("results/processed/outside/%d_%d.csv"%(iter_i,rep_j),index=None)
