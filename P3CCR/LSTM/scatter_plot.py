import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import tensorflow as tf
import plotly.express as px


filename = 'lsmt.csv'

data = pd.read_csv(filename, names=["num_ensayo", "num_capas", "num_units", "dropout", "window", "optimizer", "lrate",
                                    "batch_size", "epochs", "MAE", "Loss (MSE)"])


fig, axes = plt.subplots(nrows=2, ncols=5, sharex='col',
                         sharey='row', constrained_layout=True)
#fig.suptitle('Scatter plot hiperparámetros vs Loss y MAE')
axes[0][0].scatter(data["num_capas"], data["MAE"],
                   c=data["num_ensayo"], s=30, edgecolors="black")
axes[0][0].set_ylabel("MAE", fontsize=15)
axes[0][0].tick_params(axis = 'both', which = 'major', labelsize = 15)
axes[0][1].scatter(data["num_units"], data["MAE"],
                   c=data["num_ensayo"], s=30, edgecolors="black")
axes[0][2].scatter(data["window"], data["MAE"],
                c=data["num_ensayo"], s=30, edgecolors="black")
axes[0][3].scatter(data["lrate"], data["MAE"],
                   c=data["num_ensayo"], s=30, edgecolors="black")
axes[0][4].scatter(data["batch_size"], data["MAE"],
                   c=data["num_ensayo"], s=30, edgecolors="black")

axes[1][0].scatter(data["num_capas"], data["Loss (MSE)"],
                   c=data["num_ensayo"], s=30, edgecolors="black")
axes[1][0].set_xlabel("Nº de capas", fontsize=15)
axes[1][0].set_ylabel("Loss (MSE)", fontsize=15)
axes[1][0].tick_params(axis = 'both', which = 'major', labelsize = 15)

axes[1][1].scatter(data["num_units"], data["Loss (MSE)"],
                   c=data["num_ensayo"], s=30, edgecolors="black")
axes[1][1].set_xlabel("Nº de neuronas", fontsize=15)
axes[1][1].tick_params(axis = 'both', which = 'major', labelsize = 15)

axes[1][2].scatter(data["window"], data["Loss (MSE)"],
                   c=data["num_ensayo"], s=30, edgecolors="black")
axes[1][2].set_xlabel("Window", fontsize=15)
axes[1][2].tick_params(axis = 'both', which = 'major', labelsize = 15)

axes[1][3].scatter(data["lrate"], data["Loss (MSE)"],
                   c=data["num_ensayo"], s=30, edgecolors="black")
axes[1][3].set_xlabel("L. Rate", fontsize=15)
axes[1][3].tick_params(axis = 'both', which = 'major', labelsize = 15)
axes[1][3].set_xlim([0, 0.0011])


axes[1][4].scatter(data["batch_size"], data["Loss (MSE)"],
                   c=data["num_ensayo"], s=30, edgecolors="black")
axes[1][4].set_xlabel("Batch size", fontsize=15)
axes[1][4].tick_params(axis = 'both', which = 'major', labelsize = 15)
cax, kw = mpl.colorbar.make_axes([ax for ax in axes.flat])
clb = plt.colorbar(axes[0][0].get_children()[0], cax=cax, **kw, aspect=10)
clb.ax.set_title("Nº Ensayo")
plt.colorbar(axes[0][0].get_children()[0], cax=cax, **kw)
plt.show()


fig = px.parallel_coordinates(data, color="RMSE",
                              dimensions=["num_units", "dropout", "window", "optimizer", "lrate",
                                          "batch_size", "epochs", "RMSE", "Loss"],
                              color_continuous_scale=px.colors.diverging.Tealrose,
                              color_continuous_midpoint=0.92)
# fig.show()
