#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 21:10:19 2023

@author: orel
"""

import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.mixture import GaussianMixture
import itertools

color_iter = itertools.cycle(["navy", "c", "cornflowerblue", "gold", "darkorange"])

def plot_results(X, Y_, means, covariances, title):
    fig, ax = plt.subplots()
    for i, (mean, covar, color) in enumerate(zip(means, covariances, color_iter)):
        v, w = np.linalg.eigh(covar)
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        u = w[0] / np.linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], 0.8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180.0 * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], angle=180.0 + angle, color=color)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)

    plt.xlim(-9.0, 5.0)
    plt.ylim(-3.0, 6.0)
    plt.xticks(())
    plt.yticks(())
    plt.title(title)
    

from PIL import Image
import cv2
import time
import matplotlib.pyplot as plt

kernel = [30, 50, 75, 100]; 
sigma = [30, 50, 75, 100];

img = 255 - np.asarray(Image.open("/home/orel/orel_ws/thesis/proof of concept/maps/map1.png"))
pos = (50, 350)
end = (350, 225)
cost_map = np.zeros(img.shape)
xx = np.arange(img.shape[1])[None, :]
yy = np.arange(img.shape[0])[:, None]
grid_position = pos
rr = np.sqrt((xx - grid_position[0])**2 + (yy - grid_position[1])**2)

rr = rr / np.max(rr)

walls = np.any(img > 255 * .9, axis=2)
walls = cv2.filter2D(walls.astype(np.uint8), -1, np.ones((5,5)))
walls[walls > 1] = 1


gg = cv2.getGaussianKernel(50, 30)
gg_ = gg * gg.T
obs = cv2.filter2D(walls.astype(np.uint8), 5, gg_)
obs = obs / np.max(obs)
cost_map = 0.5 * obs + 0.5 * rr
cost_map[walls > 0] = 1
grad = np.gradient(cost_map)
cost_map = (cost_map * 255).astype(np.uint8)
cv2.imshow("b", cost_map)

cost_map = np.tile(cost_map[..., None], 3)

path = np.array([end])
while np.linalg.norm(path[-1,:] - pos) > 10 and path.shape[0] < 10_000:
    dx = -1 + 2 *(grad[1][path[-1,1], path[-1,0]] > 0)
    dy = -1 + 2 *(grad[0][path[-1,1], path[-1,0]] > 0)
    
    next_pos = np.array((path[-1,0] - dx, path[-1,1] - dy))
    cost_map = cv2.circle(cost_map, next_pos, radius=1, color=(255,0,0), thickness=-1)
    path = np.r_[path, next_pos[None, :]]


cost_map = cv2.circle(cost_map, pos, radius=5, color=(0,0,255), thickness=-1)

cost_map = cv2.circle(cost_map, end, radius=5, color=(0,255,0), thickness=-1)

cv2.imshow("mm", cost_map)


# for k in kernel:
#     st = time.time()
#     for s in sigma:
        
#         gg = cv2.getGaussianKernel(k, s)
#         gg_ = gg * gg.T
#         gg_ = gg_
#         f_img = cv2.filter2D(walls.astype(np.uint8), 5, gg_)
#         f_img = f_img / np.max(f_img) * 255
#         f_img[walls] = 255
#         cv2.imshow(f"k:{k} sigma:{s}", f_img.astype(np.uint8))
#     times.append((time.time() - st))
        

k = 50
M, N = cost_map.shape
pad = ((0, M % k), (0, N % k))

M, N = cost_map.shape
cost_map = np.pad(cost_map, pad)

m = M // k
n = N // k

reduce_cm = cost_map[:m*k, :n*k].reshape(m, k, n, k).max(axis=(1,3))

mat = np.array([[20,  200,   -5,   23, 7],
                [-13,  134,  119,  100, 8],
                [120,   32,   49,   25, 12],
                [-120,   12,   9,   23, 15],
                [-57,   84,   19,   17, 82],
                ])
# soln
# [200, 119, 8]
# [120, 49, 15]
# [84, 19, 82]
M, N = mat.shape
K = 2
L = 2

MK = M // K
NL = N // L



# split the matrix into 'quadrants'
Q1 = mat[:MK * K, :NL * L].reshape(MK, K, NL, L).max(axis=(1, 3))
Q2 = mat[MK * K:, :NL * L].reshape(-1, NL, L).max(axis=2)
Q3 = mat[:MK * K, NL * L:].reshape(MK, K, -1).max(axis=1)
Q4 = mat[MK * K:, NL * L:].max()

# compose the individual quadrants into one new matrix
soln = np.vstack([np.c_[Q1, Q3], np.c_[Q2, Q4]])
print(soln)
# [[200 119   8]
#  [120  49  15]
#  [ 84  19  82]]



n_map = cost_map[:m]