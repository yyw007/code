import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from scipy.interpolate import griddata
from matplotlib import cm
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
# Load the data from Excel files
experimental_space_data = pd.read_excel(r"Experimental Space.xlsx")
design_results_data = pd.read_excel(r"Design results excellent 5.xlsx")

# Extracting the 'Success or Fail' column
success_fail = experimental_space_data['Success or Fail']

# The columns to perform t-SNE
columns = ['Temperature', 'Speed', 'Spray Flow', 'Plamsa Height', 'Plasma Gas Flow', 'Plasma DC']

# Perform t-SNE on experimental space data
tsne = TSNE(n_components=3, random_state=42, perplexity=3)
tsne_results_exp = tsne.fit_transform(experimental_space_data[columns])

# Perform t-SNE on design results data
tsne_results_design = tsne.fit_transform(design_results_data[columns])

# Prepare the axes for the 3D plot from the t-SNE results
x_exp = tsne_results_exp[:, 0]
y_exp = tsne_results_exp[:, 1]
z_exp = tsne_results_exp[:, 2]

# Interpolation for a smoother surface
xi, yi = np.meshgrid(np.linspace(x_exp.min(), x_exp.max(), 100), np.linspace(y_exp.min(), y_exp.max(), 100))
zi = griddata((x_exp, y_exp), z_exp, (xi, yi), method='cubic')

# Interpolating 'Success or Fail' for color mapping to the new grid size
success_fail_colors = griddata((x_exp, y_exp), success_fail, (xi, yi), method='nearest')

# Define a simple two-color colormap
cmap = ListedColormap(['gold', 'mediumorchid'])  # gold for Fail (0), mediumorchid for Success (1)

# Create the 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plotting the surface with two colors based on 'Success or Fail'
surface = ax.plot_surface(xi, yi, zi, facecolors=cmap(success_fail_colors), alpha=0.6, edgecolor='none')

# Plotting the design results data points
ax.scatter(tsne_results_design[:, 0], tsne_results_design[:, 1], tsne_results_design[:, 2], color='red', label='Design Results', s=50, alpha=1)

ax.set_xlabel('t-SNE Feature 1', fontsize=13)
ax.set_ylabel('t-SNE Feature 2', fontsize=12)
ax.set_zlabel('t-SNE Feature 3', fontsize=13)

plt.title('')
plt.legend(loc='upper right',fontsize='x-large')
# 添加图例
legend_lines = [Line2D([0], [0], color='purple', label='Virtual Experimental Space'),
                Line2D([0], [0], color='red',marker='o', linestyle='None', label='Optimized Results')]
ax.legend(handles=legend_lines, fontsize=16, loc=(0.5, 0.95))

plt.tick_params(axis='x', labelsize=11)  # X轴刻度字体大小
plt.tick_params(axis='y', labelsize=11)  # Y轴刻度字体大小

plt.colorbar(surface, ax=ax, shrink=0.5, aspect=5, ticks=[0, 1], label='Success or Fail')  # Adding a colorbar for the surface


plt.show()
