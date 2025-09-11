import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parameters
n_cells_to_spread = 4
neighbor_range = np.arange(-n_cells_to_spread, n_cells_to_spread + 1)
title_fontsize = 20
label_fontsize = 14
tick_fontsize = 12

# Build coordinate grids and compute distances
dx, dy, dz = np.meshgrid(
    neighbor_range, neighbor_range, neighbor_range, indexing='ij'
)
coords = np.stack([dx.ravel(), dy.ravel(), dz.ravel()], axis=1)
dists = np.linalg.norm(coords, axis=1)

# Create a boolean mask for voxels inside the sphere
mask = dists <= n_cells_to_spread
sphere_voxels = mask.reshape((2 * n_cells_to_spread + 1,) * 3)

# Prepare facecolors: orange for all, red for center
facecolors_full = np.zeros(sphere_voxels.shape + (4,), dtype=float)
facecolors_full[..., :] = (1.0, 0.65, 0.0, 1.0)  # Orange
center = n_cells_to_spread
facecolors_full[center, center, center] = (1.0, 0.0, 0.0, 1.0)  # Red

# Create half-sphere mask
half_voxels = sphere_voxels.copy()
half_voxels[dy < 0] = False
facecolors_half = facecolors_full.copy()

# Plot side by side
fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')

# Full sphere on left
ax1.voxels(sphere_voxels, facecolors=facecolors_full,
           edgecolor='k', linewidth=0.3)
ax1.set_title('Neighboring cells', fontsize=title_fontsize)
ax1.set_xlabel('dx', labelpad=10, fontsize=label_fontsize)
ax1.set_ylabel('dy', labelpad=10, fontsize=label_fontsize)
ax1.set_zlabel('dz', labelpad=20, fontsize=label_fontsize)
ax1.zaxis._axinfo['label']['space_factor'] = 2.5
ax1.set_xticks(np.arange(len(neighbor_range)))
ax1.set_xticklabels([str(v) for v in neighbor_range], fontsize=tick_fontsize)
ax1.set_yticks(np.arange(len(neighbor_range)))
ax1.set_yticklabels([str(v) for v in neighbor_range], fontsize=tick_fontsize)
ax1.set_zticks(np.arange(len(neighbor_range)))
ax1.set_zticklabels([str(v) for v in neighbor_range], fontsize=tick_fontsize)
ax1.set_box_aspect([1, 1, 1])

# Half sphere on right
ax2.voxels(half_voxels, facecolors=facecolors_half,
           edgecolor='k', linewidth=0.3)
ax2.set_title('Neighboring cells (Half Sphere)', fontsize=title_fontsize)
ax2.set_xlabel('dx', labelpad=10, fontsize=label_fontsize)
ax2.set_ylabel('dy', labelpad=10, fontsize=label_fontsize)
ax2.set_zlabel('dz', labelpad=20, fontsize=label_fontsize)
ax2.zaxis._axinfo['label']['space_factor'] = 2.5
ax2.set_xticks(np.arange(len(neighbor_range)))
ax2.set_xticklabels([str(v) for v in neighbor_range], fontsize=tick_fontsize)
ax2.set_yticks(np.arange(len(neighbor_range)))
ax2.set_yticklabels([str(v) for v in neighbor_range], fontsize=tick_fontsize)
ax2.set_zticks(np.arange(len(neighbor_range)))
ax2.set_zticklabels([str(v) for v in neighbor_range], fontsize=tick_fontsize)
ax2.set_box_aspect([1, 1, 1])

plt.tight_layout()
plt.show()
