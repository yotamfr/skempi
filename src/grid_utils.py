import math
import numpy as np
from functools import reduce
from scipy.linalg import expm, norm
from scipy.ndimage.filters import gaussian_filter

from aaindex import *

VDW = {'C': 1.7, 'O': 1.52, 'N': 1.55, 'S': 1.8}


def ones(struct, atom_types):
    return np.asarray([1 for a in struct.atoms if a.type in atom_types])


def get_4channel_voxels(struct, res, R, nv=20):
    c1 = get_3d_voxels_around_res(struct, ones, res, rot=R, num_voxels=nv, atom_types=['C'])
    c2 = get_3d_voxels_around_res(struct, ones, res, rot=R, num_voxels=nv, atom_types=['N'])
    c3 = get_3d_voxels_around_res(struct, ones, res, rot=R, num_voxels=nv, atom_types=['O'])
    c4 = get_3d_voxels_around_res(struct, ones, res, rot=R, num_voxels=nv, atom_types=['S'])
    return np.stack((c1, c2, c3, c4), axis=0)


def hydrophobicity(struct, atom_types):
    return np.asarray([ARGP820101[a.res.name] for a in struct.atoms if a.type in atom_types])
def molweight(struct, atom_types):
    return np.asarray([FASG760101[a.res.name] for a in struct.atoms if a.type in atom_types])
def bfactor(struct, atom_types):
    return np.asarray([a.temp for a in struct.atoms if a.type in atom_types])


def get_feature_maps(struct, res, R, nv=20):
    ch = get_3d_voxels_around_res(struct, hydrophobicity, res, rot=R, num_voxels=nv, atom_types=['C', 'N', 'O', 'S'])
    cm = get_3d_voxels_around_res(struct, molweight, res, rot=R, num_voxels=nv, atom_types=['C', 'N', 'O', 'S'])
    cb = get_3d_voxels_around_res(struct, bfactor, res, rot=R, num_voxels=nv, atom_types=['C', 'N', 'O', 'S'])
    return np.stack([ch, cm, cb], axis=0)


def get_voxels(struct, res, R, nv=20):
    ca = get_4channel_voxels(struct, res, R, nv)
    cf = get_feature_maps(struct, res, R, nv)
    return ca, cf


def get_3d_voxels_around_res(struct, values_from_struct, res, rot=None, num_voxels=20, atom_types=['C'], rad=0.0):
    smooth = rad/3.0
    center = res.ca.coord
    r = num_voxels // 2
    X = np.asarray([a.coord for a in struct.atoms if a.type in atom_types])
    y = values_from_struct(struct, atom_types)
    if len(X) == 0:
        return voxelize([], [], num_voxels, smooth=smooth)
    X -= center
    if rot is not None:
        X = rot(X)
    x_indx = np.intersect1d(np.where(X[:, 0] > -r), np.where(X[:, 0] < r))
    y_indx = np.intersect1d(np.where(X[:, 1] > -r), np.where(X[:, 1] < r))
    z_indx = np.intersect1d(np.where(X[:, 2] > -r), np.where(X[:, 2] < r))
    indx = reduce(np.intersect1d, [x_indx, y_indx, z_indx])
    return voxelize(X[indx, :], y[indx], num_voxels, smooth=smooth)


def rot(axis, theta):
    return expm(np.cross(np.eye(3), axis/norm(axis)*theta))


def rot_x(theta):
    return Rotation([1, 0, 0], theta)


def rot_y(theta):
    return Rotation([0, 1, 0], theta)


def rot_z(theta):
    return Rotation([0, 0, 1], theta)


class Rotation(object):

    def __init__(self, axis, theta):
        self.axis = axis
        self.theta = theta

    @property
    def R(self):
        axis = self.axis
        theta = self.theta
        return rot(axis, theta)

    def __call__(self, X):
        ret = np.dot(self.R, X.T).T
        return ret

    def __hash__(self):
        axis = self.axis
        theta = self.theta
        return hash((tuple(axis), theta))


def voxelize(X, y, num_voxels, smooth=0.0):
    n = num_voxels
    voxels = np.zeros((n + 1, n + 1, n + 1))
    for coord, val in zip(X, y):
        i, j, k = np.floor(coord + n / 2)
        voxels[int(i), int(j), int(k)] = val
    if smooth != 0.0:
        voxels[:, :, :] = gaussian_filter(voxels, sigma=smooth, order=0, output=None, mode='reflect', cval=0.0, truncate=4.0)
    return voxels


class View(object):

    def __init__(self, struct, center_res, num_voxels=20, atom_types=['C', 'O', 'N', 'S']):
        center = center_res.center
        r = num_voxels // 2
        X = np.asarray([a.coord for a in struct.atoms if a.type in atom_types])
        Y = np.asarray([a.type for a in struct.atoms if a.type in atom_types])
        X -= center
        indx = np.where(X[:, 0]**2 + X[:, 1]**2 + X[:, 2]**2 <= r**2)
        self.S = X[indx, :][0]
        self.types = Y[indx]
        self.atom_types = atom_types
        self.nv = num_voxels

    def rotate(self, R):
        self.S = R(self.S)

    @property
    def voxels(self):
        channels = [voxelize(self.S[self.types == c, :], self.nv, True) for c in self.atom_types]
        return np.stack(channels, axis=0)


if __name__ == "__main__":
    pass