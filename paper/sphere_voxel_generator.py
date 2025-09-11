import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def parse_args():
    parser = argparse.ArgumentParser(
        description='Render a voxel sphere (optionally half-sphere) with center highlight.'
    )
    parser.add_argument(
        '--half',
        action='store_true',
        help='If set, cut the sphere along the z–x plane (keep y ≥ 0).'
    )
    return parser.parse_args()


def build_sphere(n):
    # Create grid coordinates
    rng = np.arange(-n, n + 1)
    dx, dy, dz = np.meshgrid(rng, rng, rng, indexing='ij')
    coords = np.stack([dx.ravel(), dy.ravel(), dz.ravel()], axis=1)
    dists = np.linalg.norm(coords, axis=1)

    # Mask for inside-sphere voxels
    mask = dists <= n
    sphere = mask.reshape((2*n+1,)*3)
    return sphere, dx, dy, dz, rng


def plot_voxels(sphere_voxels, dx, dy, dz, rng, half=False):
    # Optionally slice half-sphere
    voxels = sphere_voxels.copy()
    if half:
        voxels[dy < 0] = False

    title_fontsize = 20
    label_fontsize = 16
    tick_fontsize = 12

    # Prepare colors: orange for all, red for center
    facecolors = np.zeros(voxels.shape + (4,), dtype=float)
    facecolors[..., :] = (1.0, 0.65, 0.0, 1.0)  # Orange
    center = voxels.shape[0] // 2
    facecolors[center, center, center] = (1.0, 0.0, 0.0, 1.0)  # Red

    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(voxels, facecolors=facecolors, edgecolor='k', linewidth=0.3)

    # Axis labels
    ax.set_xlabel('dx', labelpad=10, fontsize=label_fontsize)
    ax.set_ylabel('dy', labelpad=10, fontsize=label_fontsize)
    ax.set_zlabel('dz', labelpad=20, fontsize=label_fontsize)
    ax.zaxis._axinfo['label']['space_factor'] = 2.5

    # Set tick positions and labels to match coordinate values
    ticks = np.arange(len(rng))
    labels = [str(val) for val in rng]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=tick_fontsize)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=tick_fontsize)
    ax.set_zticks(ticks)
    ax.set_zticklabels(labels, fontsize=tick_fontsize)

    # Title
    title = 'Neighboring cells'
    if half:
        title += ' (half-sphere)'
    ax.set_title(title, fontsize=title_fontsize)
    ax.set_box_aspect([1, 1, 1])
    plt.tight_layout()
    plt.show()


def main():
    args = parse_args()
    n = 4
    sphere, dx, dy, dz, rng = build_sphere(n)
    plot_voxels(sphere, dx, dy, dz, rng, half=args.half)


if __name__ == '__main__':
    main()
