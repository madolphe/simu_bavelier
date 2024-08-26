import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import re

def plot_hypercube_subplots(hypercube, path="./", name="competence", episode_number=0, vmin=0, vmax=1):
    dims = hypercube.shape
    n_dims = len(dims)

    # Create subplots for each pair (dim1, dimX) where X varies from 2 to N
    fig, axes = plt.subplots(1, n_dims - 1, figsize=(5 * (n_dims - 1), 5))

    # Handle case where there's only one subplot (special case for N = 3)
    if n_dims - 1 == 1:
        axes = [axes]

    for i in range(1, n_dims):
        # Select slice of the hypercube using dim1 as a reference and varying dimX
        grid = np.sum(hypercube, axis=tuple(j for j in range(n_dims) if j != 0 and j != i))

        # Plot the grid with square cells
        ax = axes[i - 1]
        cax = ax.imshow(grid, cmap='viridis', interpolation='none', aspect='equal', vmin=vmin, vmax=vmax)
        ax.set_title(f'Dim1 vs Dim{i + 1} - ep {episode_number}')
        ax.set_xlabel(f'Dim{i + 1}')
        ax.set_ylabel('Dim1')

        # Customize the gridlines to be aligned with the cells
        ax.set_xticks(np.arange(0.5, grid.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(0.5, grid.shape[0], 1), minor=True)
        ax.grid(True, which='minor', color='black', linestyle='-', linewidth=2)
        ax.tick_params(which='minor', size=0)

        # Add color bar
        fig.colorbar(cax, ax=ax)

    # Adjust layout
    plt.tight_layout()
    plt.savefig(f"{path}/{name}")
    plt.close()


def create_gif_from_images(image_folder, output_gif, duration=500):
    # Extract numeric parts from file names and sort by them
    image_files = sorted(
        [os.path.join(image_folder, file) for file in os.listdir(image_folder) if file.endswith('.png')],
        key=lambda x: int(re.findall(r'\d+', os.path.basename(x))[0]))

    # Load all images into a list
    images = [Image.open(image_file) for image_file in image_files]

    # Save as a GIF
    images[0].save(output_gif, save_all=True, append_images=images[1:], duration=duration, loop=0)


def plot_trajectory(time_index, values, path, name):
    plt.plot(time_index, values)
    plt.savefig(f"{path}/{name}.png")
    plt.close()
