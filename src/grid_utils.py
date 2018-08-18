import math
import numpy as np
from functools import reduce
from scipy.linalg import expm, norm
from scipy.ndimage.filters import gaussian_filter


def get_4channel_voxels(struct, res,  nv=20, R=np.eye(3)):
    c1 = get_3d_grid_around_res(struct, res, rot_mat=R, num_voxels=nv, atom_types=['C']).voxels
    c2 = get_3d_grid_around_res(struct, res, rot_mat=R, num_voxels=nv, atom_types=['O']).voxels
    c3 = get_3d_grid_around_res(struct, res, rot_mat=R, num_voxels=nv, atom_types=['N']).voxels
    c4 = get_3d_grid_around_res(struct, res, rot_mat=R, num_voxels=nv, atom_types=['S']).voxels
    return np.stack((c1, c2, c3, c4), axis=0)


def get_3d_grid_around_res(struct, res, rot_mat=np.eye(3), num_voxels=20, atom_types=['C']):
    center = res.ca.coord
    r = num_voxels // 2
    X = np.asarray([a.coord for a in struct.atoms if a.type in atom_types]) - center
    X = np.dot(rot_mat, X.T).T
    x_indx = np.intersect1d(np.where(X[:, 0] > -r), np.where(X[:, 0] < r))
    y_indx = np.intersect1d(np.where(X[:, 1] > -r), np.where(X[:, 1] < r))
    z_indx = np.intersect1d(np.where(X[:, 2] > -r), np.where(X[:, 2] < r))
    indx = reduce(np.intersect1d, [x_indx, y_indx, z_indx])
    return Grid(X[indx, :], num_voxels)


def rot(axis, theta):
    return expm(np.cross(np.eye(3), axis/norm(axis)*theta))


def rot_x(theta):
    ret = np.eye(3)
    ret[1, 1] = np.cos(theta)
    ret[2, 2] = np.cos(theta)
    ret[2, 1] = np.sin(theta)
    ret[1, 2] = -np.sin(theta)
    return ret


def rot_y(theta):
    ret = np.eye(3)
    ret[0, 0] = np.cos(theta)
    ret[2, 2] = np.cos(theta)
    ret[0, 2] = np.sin(theta)
    ret[2, 0] = -np.sin(theta)
    return ret


def rot_z(theta):
    ret = np.eye(3)
    ret[0, 0] = np.cos(theta)
    ret[1, 1] = np.cos(theta)
    ret[1, 0] = np.sin(theta)
    ret[0, 1] = -np.sin(theta)
    return ret


class Grid(object):
    def __init__(self, X, num_voxels=20):
        self.X = X
        self.num_voxels = num_voxels

    @property
    def voxels(self, smooth=True):
        n = self.num_voxels
        voxels = np.zeros((n+1, n+1, n+1))
        for coord in self.X:
            i, j, k = np.floor(coord + n / 2)
            voxels[int(i), int(j), int(k)] = 1
        if smooth:
            voxels[:, :, :] = gaussian_filter(voxels, sigma=0.6, order=0, output=None, mode='reflect', cval=0.0, truncate=4.0)
            voxels[:, :, :] *= 1000
        return voxels


if __name__ == "__main__":
    pass