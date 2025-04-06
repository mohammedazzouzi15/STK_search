# Load an existing figure from a file
# Replace 'example_figure.png' with the path to your saved figure
# Note: Matplotlib does not directly load figures from image files.
# If you have the figure saved as a `.pickle` file, you can load it.
import pickle

import numpy as np
from matplotlib.ticker import MaxNLocator
name = "fig_runs10.pkl"
with open(name, "rb") as f:
    fig = pickle.load(f)

fig.axes[1].set_xlabel("")
fig.axes[3].set_xlabel("")
fig.axes[1].set_ylabel("")
fig.axes[3].set_ylabel("")
# Hide tick labels for specific axes
fig.axes[1].tick_params(labelleft=False)  # Hide bottom and left tick labels
fig.axes[3].tick_params(labelleft=False)  # Hide bottom and left tick labels

fig.axes[1].set_frame_on(False)
fig.axes[3].set_frame_on(False)
min_num_iteration = 810
x_limits = {
    0: (50, min_num_iteration),
    2: (50, min_num_iteration),
    4: (50, min_num_iteration),
    5: (50, min_num_iteration),
    1: (0, 100),
    3: (0, 3100),
}
y_limits = {
    0: (0, 0.7),
    1: (0, 0.7),
    2: (-2, 0),
    3: (-2, 0),
    4: (0, 65),
    5: (0, 0.63),
}
tick_labels = {
    4: np.arange(0, min_num_iteration, 100),
    5: np.arange(0, min_num_iteration, 100),
    3: [0, 15, 30],
}
fig.axes[4].xaxis.set_major_locator(MaxNLocator(nbins=8))
fig.axes[5].xaxis.set_major_locator(MaxNLocator(nbins=8))
fig.axes[0].xaxis.set_major_locator(MaxNLocator(nbins=8))
fig.axes[2].xaxis.set_major_locator(MaxNLocator(nbins=8))
fig.axes[1].xaxis.set_major_locator(MaxNLocator(nbins=3))
fig.axes[3].xaxis.set_major_locator(MaxNLocator(nbins=3))
# Apply axis limits and tick labels
for ax_index, ax in enumerate(fig.axes):
    if ax_index in x_limits:
        ax.set_xlim(*x_limits[ax_index])
    if ax_index in y_limits:
        ax.set_ylim(*y_limits[ax_index])
    if ax_index in tick_labels:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=len(tick_labels[ax_index])))
        ax.xaxis.set_ticklabels(tick_labels[ax_index])
legend_list = [
        "BO-learned",
        #"BO-Mord",
        "BO-Prop",
        "SUEA",
        "EA",
        "Rand",
    ]
h,l = fig.axes[0].get_legend_handles_labels()
# Modify the legend
lg = fig.axes[0].legend(h,
    legend_list,
    loc="upper left",  # Change the location of the legend
    bbox_to_anchor=(0.1, 1.2),  # Adjust the position of the legend
    ncol=6,  # Number of columns in the legend
    fontsize=20,  # Font size of the legend text

    frameon=True,  # Add a frame around the legend
    shadow=False,  # Add a shadow to the legend
    fancybox=True,  # Use a rounded box for the legend
    edgecolor="black",  # Set the edge color of the legend box
)



fig.tight_layout()
fig.savefig("mod_"+name.replace(".pkl", ".png"))

# Optionally, save the modified figure as a `.pickle` file for further editing
with open("mod_"+name, "wb") as f:
    pickle.dump(fig, f)

# Show the modified figure
