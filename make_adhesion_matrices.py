import json
import numpy as np
import sys
import os

if not os.path.exists("adhesion_matrices"):
    os.mkdir("adhesion_matrices")

# n_iter = int(sys.argv[1])
n_iter = 100

adhesion_dict = json.load(open("raw_data/adhesion_dict.json"))

nE,nT,nX = 12,20,10

pair_names = {(0,0):"ES-ES",
              (1,1):"TS-TS",
              (0,1):"TS-ES",
              (1,0):"TS-ES",
              (0,2):"XEN-ES",
              (2,0):"XEN-ES",
              (1,2):"XEN-TS",
              (2,1):"XEN-TS",
              (2,2):"XEN-XEN"}

def sample(pair):
    val = np.nan
    while np.isnan(val):
        val = np.random.choice(adhesion_dict[pair_names[pair]])
    return val

def get_adhesion_matrix(nE,nT,nX):
    nc = nE + nT + nX

    c_types = np.zeros((nc),dtype=int)
    c_types[nE:nE+nT] = 1
    c_types[nE+nT:] = 2

    c_type1,c_type2 = np.meshgrid(c_types,c_types,indexing="ij")
    c_type_pairs = list(zip(c_type1.ravel(),c_type2.ravel()))
    adhesion_vals = list(map(sample,c_type_pairs))
    adhesion_vals = np.array(adhesion_vals).reshape((len(c_types),len(c_types)))
    adhesion_vals = np.triu(adhesion_vals,1) + np.triu(adhesion_vals,1).T
    return adhesion_vals

nc = nE + nT + nX
for i in range(n_iter):
    adhesion_vals = get_adhesion_matrix(nE,nT,nX)
    adhesion_vals_full = np.zeros((nc+8,nc+8))
    adhesion_vals_full[8:,8:] = adhesion_vals
    np.savez("adhesion_matrices/%i.npz"%i,adhesion_vals=adhesion_vals)
