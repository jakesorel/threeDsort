from scipy.sparse import coo_matrix
import numpy as np

def assemble_tet(val,tet,nc):
    return coo_matrix((val.ravel(),(tet.ravel(),np.zeros(tet.size,dtype=int))),shape=(nc,1)).toarray()

def assemble_tet3(val3,tet,nc):
    return coo_matrix((val3.ravel(),(np.repeat(tet.ravel(),3),np.tile(np.arange(3),tet.size))),shape=(nc,3)).toarray()

