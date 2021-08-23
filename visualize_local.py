import h5py
import numpy as np
import os, sys
import open3d as o3d
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import time


def axisEqual3D(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = 0.15  # maxsize / 2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)


def visualize_points(points, n_particles):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # import pdb; pdb.set_trace()
    # green is the first thing in positions
    # red is all the neighbors
    # shaded is all the particles
    # blue is the ball
    #     ax.scatter(queries[-1, 0], queries[-1, 1], queries[-1, 2], c='b', s=80)
    # ax.scatter(queries[idx, 0], queries[idx, 1], queries[idx, 2], c='g', s=80)
    #     ax.scatter(anchors[neighbors, 0], anchors[neighbors, 1], anchors[neighbors, 2], c='r', s=80)
    #     ax.scatter(anchors[:, 0], anchors[:, 1], anchors[:, 2], alpha=0.2)
    if task_name == "Gripper":
        #         import pdb; pdb.set_trace()
        ax.scatter(points[-1, 0], points[-1, 1], points[-1, 2], c='b', s=200)
        ax.scatter(points[-2, 0], points[-2, 1], points[-2, 2], c='b', s=200)
        ax.scatter(points[:n_particles, 0], points[:n_particles, 1], points[:n_particles, 2], alpha=0.5, s=100)
        ax.scatter(points[-3, 0], points[-3, 1], points[-3, 2], c='g', s=200)
    else:
        ax.scatter(points[-1, 0], points[-1, 1], points[-1, 2], c='b', s=200)
        ax.scatter(points[:n_particles, 0], points[:n_particles, 1], points[:n_particles, 2], alpha=0.5, s=100)
        ax.scatter(points[-2, 0], points[-2, 1], points[-2, 2], c='g', s=200)
    axisEqual3D(ax)

    plt.show()

def load_data(data_names, path):
    hf = h5py.File(path, 'r')
    data = []
    for i in range(len(data_names)):
        d = np.array(hf.get(data_names[i]))
        data.append(d)
    hf.close()
    return data


task_name = "Gripper"
rollout_dir = f"./data/data_{task_name}_bak/fps/"
n_vid = 1
n_frame = 49
data_names = ['positions', 'shape_quats', 'scene_params']
counts = 0
counts_d = 0
sum_p = np.zeros(3)
sum_d = np.zeros(3)
start_frame = 0
for i in range(1):
    for t in range(start_frame, start_frame+n_frame):
        print(f"visualizing {t}")
        if task_name == "Gripper":
            frame_path = os.path.join(rollout_dir, str(t) + '.h5')
        else:
            frame_path = os.path.join(rollout_dir, 'train', str(i).zfill(3), str(t) + '.h5')
        this_data = load_data(data_names, frame_path)
        states = this_data[0]
        print(states.shape)
        states[:,[1, 2]] = states[:,[2, 1]]
#         states_n = states[:,[2, 1]]
        visualize_points(states, 300)
        # import pdb; pdb.set_trace()


