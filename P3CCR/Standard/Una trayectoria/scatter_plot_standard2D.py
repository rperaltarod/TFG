import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import tensorflow as tf
import plotly.express as px


filename = 'standard_2.csv'

data = pd.read_csv(filename, names=["num_ensayo", "num_capas","num_units","activation_fuction", "optimizer", "lrate",
                                    "batch_size", "epochs", "MAE", "Loss (MSE)"])

print(data)

fig, axes = plt.subplots(nrows=2, ncols=4, sharex='col',
                         sharey='row', constrained_layout=True, figsize=(10, 6))
plt.subplots_adjust(left=0.12, bottom=0.1, right=1.07, top=0.9, wspace=0.4, hspace=None)
#fig.suptitle('Scatter plot hiperparámetros vs Loss y MAE')
axes[0][0].scatter(data["num_capas"], data["MAE"],
                   c=data["num_ensayo"], s=30, edgecolors="black")
axes[0][0].set_ylabel("MAE", fontsize=20)
axes[0][0].tick_params(axis = 'both', which = 'major', labelsize = 15)
axes[0][1].scatter(data["num_units"], data["MAE"],
                   c=data["num_ensayo"], s=30, edgecolors="black")
axes[0][2].scatter(data["lrate"], data["MAE"],
                   c=data["num_ensayo"], s=30, edgecolors="black")
axes[0][3].scatter(data["batch_size"], data["MAE"],
                   c=data["num_ensayo"], s=30, edgecolors="black")
axes[1][0].scatter(data["num_capas"], data["Loss (MSE)"],
                   c=data["num_ensayo"], s=30, edgecolors="black")
axes[1][0].set_xlabel("Nº Capas", fontsize=20)
axes[1][0].set_ylabel("Loss (MSE)", fontsize=20)
axes[1][0].tick_params(axis = 'both', which = 'major', labelsize = 15)
axes[1][0].set_xticks([3,6,9])

axes[1][1].scatter(data["num_units"], data["Loss (MSE)"],
                   c=data["num_ensayo"], s=30, edgecolors="black")
axes[1][1].set_xlabel("Nº Neuronas", fontsize=20)
axes[1][1].tick_params(axis = 'both', which = 'major', labelsize = 15)
axes[1][1].set_xticks([0,100,200,300])

axes[1][2].scatter(data["lrate"], data["Loss (MSE)"],
                   c=data["num_ensayo"], s=30, edgecolors="black")
axes[1][2].set_xlabel("L. Rate", fontsize=20)
axes[1][2].tick_params(axis = 'both', which = 'major', labelsize = 15)
axes[1][2].set_xticks([0.0001,0.001])

axes[1][3].scatter(data["batch_size"], data["Loss (MSE)"],
                   c=data["num_ensayo"], s=30, edgecolors="black")
axes[1][3].set_xlabel("Batch size", fontsize=20)
axes[1][3].tick_params(axis = 'both', which = 'major', labelsize = 15)
axes[1][3].set_xticks([50,100,200])

cax, kw = mpl.colorbar.make_axes([ax for ax in axes.flat])
clb = plt.colorbar(axes[0][0].get_children()[0], cax=cax, **kw, aspect=10)
clb.ax.set_title("Nº Ensayo")
plt.colorbar(axes[0][0].get_children()[0], cax=cax, **kw)
plt.savefig('scatter_plot_standard2D.png', dpi=300)
plt.show()


fig = px.parallel_coordinates(data, color="MAE",
                              dimensions=["num_units", "dropout", "window", "optimizer", "lrate",
                                          "batch_size", "epochs", "MAE", "Loss (MSE)"],
                              color_continuous_scale=px.colors.diverging.Tealrose,
                              color_continuous_midpoint=0.92)
# fig.show()
