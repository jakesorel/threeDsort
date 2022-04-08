import numpy as np
# import matplotlib.pyplot as plt
from tissue import Tissue
from force import ActiveForce
# import time
# import pandas as pd
# import plotly.io as pio
# pio.renderers.default = "browser"
# import plotly.express as px
import codecs, json


class Simulation:
    def __init__(self,tissue_params,init_params,active_params,sim_params):
        self.sim_params = sim_params
        self.t = Tissue(init_params=init_params, tissue_params=tissue_params)
        self.active = ActiveForce(self.t,active_params)

        self.t_span = None
        self.nt = None
        self.x_save = None
        self.tet_save = None

        self.initialize()

    def initialize(self):
        self.set_t_span()
        self.set_saving_matrices()

    def set_t_span(self):
        self.t_span = np.arange(0,self.sim_params["tfin"],self.sim_params["dt"])
        self.nt = self.t_span.size
        self.t_span_save = self.t_span[::self.sim_params["skip"]]
        self.nt_save = self.t_span_save.size

    def set_saving_matrices(self):
        self.x_save = np.zeros(((self.nt_save,) + self.t.mesh.x.shape))
        self.tet_save = [None]*self.nt_save

    def simulate(self):
        x = self.t.mesh.x
        k = 0
        for i in range(self.t_span.size):
            x += self.sim_params["dt"] * self.t.force.F[8:]
            aF = self.active.update_active_force(self.sim_params["dt"])
            x += self.sim_params["dt"] * aF
            if (i % self.sim_params["skip"]) == 0:
                self.x_save[k] = x.copy()
                self.tet_save[k] = self.t.mesh.tet.copy()
                k+=1
            self.t.update_x(x)
    #
    # def plotly_animate(self,n_plot=50):
    #     nc = self.t.mesh.nc
    #     cid = np.arange(nc)
    #     ctypes = self.t.c_types_real
    #     skip = int(self.t_span_save.size/n_plot)
    #     x_save_plot = self.x_save[::skip]
    #     t_span_plot = self.t_span_save[::skip]
    #
    #     df = pd.DataFrame({"x":x_save_plot[:,:,0].ravel(),"y":x_save_plot[:,:,1].ravel(),"z":x_save_plot[:,:,2].ravel(),"t":np.repeat(t_span_plot,nc),"cid":np.tile(cid,t_span_plot.size),"ctype":np.tile(ctypes,t_span_plot.size)})
    #
    #     mn = min((df["x"].min(),df["y"].min(),df["z"].min()))
    #     mx = max((df["x"].max(),df["y"].max(),df["z"].max()))
    #
    #     plot_box = [mn,mx]
    #
    #     fig = px.scatter_3d(df, x="x", y="y",z="z",animation_frame="t",color="ctype",range_x=plot_box, range_y=plot_box, range_z=plot_box,size=np.repeat(10,df.shape[0]))
    #
    #     for frame in fig.frames:
    #         frame.data[0].marker.sizeref = 4e-2
    #     fig.show()

    def save_skeleton(self, name, dir_path=""):
        """
        Save the bare-bones results to a json file
        :param name:
        :param id:
        :param dir_path:
        :param compressed:
        :return:
        """
        self.name = name
        skeleton_dict = {"c_types": self.t.c_types.tolist(),
                         "x_save": self.x_save.tolist(),
                         "tet_save": [tet.tolist() for tet in self.tet_save],
                         "t_span_save": self.t_span_save.tolist(),
                         "tissue_params": serialise_dict(self.t.tissue_params),
                         "active_params": serialise_dict(self.active.active_params),
                         "simulation_params": self.sim_params}
        json.dump(skeleton_dict, codecs.open(dir_path + "/" + self.name + "_simulation" + '.json', 'w', encoding='utf-8'),
                  separators=(',', ':'),
                  sort_keys=True,
                  indent=4)
        ### this saves the array in .json format



def serialise_dict(dictionary):
    serialised_dictionary = dictionary.copy()
    for key,value in dictionary.items():
        if type(value) is np.ndarray:
            serialised_dictionary[key] = value.tolist()
        else:
            serialised_dictionary[key] = value
    return serialised_dictionary
