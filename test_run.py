import numpy as np
import matplotlib.pyplot as plt
from simulation import Simulation
from mesh import Mesh
from scipy.spatial.distance import pdist,squareform,cdist


box = np.array(((-1,-1,-1),
                (-1,-1,2),
                (-1,2,-1),
                (2,-1,-1),
                (-1,2,2),
                (2,-1,2),
                (2,2,-1),
                (2,2,2)))*100.0


adh_EE = 1.911305
adh_ET = 0.494644
adh_TT = 2.161360
adh_EX = 0.505116
adh_TX = 0.420959
adh_XX = 0.529589
#
# adh_EE = 2
# adh_ET = 2
# adh_TT = 2
# adh_EX = 2
# adh_TX = 2
# adh_XX = 2

# ES-ES      1.911305
# TS-TS      2.161360
# XEN-XEN    0.529589
# TS-ES      0.494644
# XEN-TS     0.420959
# XEN-ES     0.505116
#
# ES-ES      1.943650
# TS-TS      2.205623
# XEN-XEN    0.557278
# TS-ES      0.579996
# XEN-TS     0.460059
# XEN-ES     0.832923

d0 = 0.7
dmax = 1

def get_rep(kA,dmax,d0):
    return d0*kA/(dmax-d0)

rep_EE = get_rep(adh_EE,dmax,d0)
rep_ET = get_rep(adh_ET,dmax,d0)
rep_EX = get_rep(adh_EX,dmax,d0)
rep_TT = get_rep(adh_TT,dmax,d0)
rep_TX = get_rep(adh_TX,dmax,d0)
rep_XX = get_rep(adh_XX,dmax,d0)

rep_EE = get_rep(adh_EE,dmax,0.6)
rep_ET = get_rep(adh_ET,dmax,0.6)
rep_EX = get_rep(adh_EX,dmax,0.6)
rep_TT = get_rep(adh_TT,dmax,0.6)
rep_TX = get_rep(adh_TX,dmax,0.6)
rep_XX = get_rep(adh_XX,dmax,0.6)

tissue_params = {"kappa_adhesion":np.array(((adh_EE,adh_ET,adh_EX,0),(adh_ET,adh_TT,adh_TX,0),(adh_EX,adh_TX,adh_XX,0),(0,0,0,0))),
                 "kappa_repulsion":np.array(((rep_EE,rep_ET,rep_EX,1),(rep_ET,rep_TT,rep_TX,1),(rep_EX,rep_TX,rep_XX,1),(1,1,1,1))),
                 "dmax":1.0}
init_params = {"n_E":12,"n_T":20,"n_X":10,"box":box,"xlim":[-1,1]}

active_params = {"v0":1,"Dr":10}
sim_params = {"dt":0.02,"tfin":400,"skip":2}

sim = Simulation(tissue_params,init_params,active_params,sim_params)
sim.simulate()
sim.save_skeleton("test","results")
sim.plotly_animate()


from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

cc_E = []
cc_T = []
for x in sim.x_save[::]:
    mesh = Mesh(x,init_params["box"])
    tet = mesh.tet
    mesh.get_edges()
    edges = mesh.edges_squareform[mesh.dist_squareform<=tissue_params["dmax"]]
    adj = coo_matrix(([True]*len(edges),(edges[:,0],edges[:,1])),shape=(mesh.nC,mesh.nC))
    adj += adj.T

    c_type = 0
    cc_E.append(connected_components(adj[(sim.t.c_types==c_type)].T[(sim.t.c_types==c_type)].T)[0])
    c_type = 1
    cc_T.append(connected_components(adj[(sim.t.c_types==c_type)].T[(sim.t.c_types==c_type)].T)[0])

x_test = sim.x_save[::16]

E_types = np.expand_dims(sim.t.c_types_real == 0,1)
nE = E_types.sum()

T_types = np.expand_dims(sim.t.c_types_real == 1,1)
nT = T_types.sum()
dcentroid = np.zeros(len(x_test))
bias = np.zeros(len(x_test))
bias_p = np.zeros(len(x_test))

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

from scipy.stats import nakagami


for i, x in enumerate(x_test):
    E_centroid = (x*E_types).sum(axis=0)/nE
    T_centroid = (x*T_types).sum(axis=0)/nT
    dcentroid[i] = np.linalg.norm(E_centroid-T_centroid)
    d_centroid_sample = np.array([sample_random_d_centroid(x, sim.t.c_types_real) for i in range(int(1e4))])
    d_centroid_i = np.linalg.norm(E_centroid-T_centroid)
    bias[i] = np.linalg.norm(E_centroid-T_centroid)/d_centroid_sample.mean()
    bias_p[i] = 1-nakagami.cdf(d_centroid_i,*nakagami.fit(d_centroid_sample))
    #
    # mesh = Mesh(x,init_params["box"])
    # D = cdist(np.array((E_centroid,T_centroid)),x)
    # dist_to_centroid = ((D[0]*(sim.t.c_types_real==0)).sum() + (D[1]*(sim.t.c_types_real==1)).sum())/(nE+nT)
    # bias[i] = (dcentroid[i]/dist_to_centroid)

fig, ax = plt.subplots()
# plt.plot(bias)
plt.plot(bias_p)
plt.show()

#
#

x_test = sim.x_save[::4]
dists = np.zeros((x_test.shape[0],4))
es_ids = np.nonzero(sim.t.c_types == 0)[0]
ts_ids = np.nonzero(sim.t.c_types == 1)[0]
xen_ids = np.nonzero(sim.t.c_types == 2)[0]
frac_outside = np.zeros((x_test.shape[0],3))
for i, x in enumerate(x_test):
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
plt.plot(frac_outside)
plt.show()

