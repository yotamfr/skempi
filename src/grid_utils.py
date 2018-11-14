import math
import numpy as np
from functools import reduce
from scipy.linalg import expm, norm
from scipy.ndimage.filters import gaussian_filter

from pdb_utils import *
from aaindex import *

VDW = {'C': 1.7, 'O': 1.52, 'N': 1.55, 'S': 1.8}

BACKBONE_ATOMS = ['CA', 'C', 'N', 'O']

CNOS = ["C", "N", "O", "S"]

MAX_DISTANCE_CUTOFF = 15.0


def ones(atoms, atom_types):
    return np.asarray([1 for a in atoms if a.type in atom_types])


def get_4channel_voxels_around_res(atoms, res, R, nv=20):
    c1 = get_3d_voxels_around_res(atoms, res, ones, rot=R, num_voxels=nv, atom_types=['C'], vdw=1.70)
    c2 = get_3d_voxels_around_res(atoms, res, ones, rot=R, num_voxels=nv, atom_types=['N'], vdw=1.55)
    c3 = get_3d_voxels_around_res(atoms, res, ones, rot=R, num_voxels=nv, atom_types=['O'], vdw=1.52)
    c4 = get_3d_voxels_around_res(atoms, res, ones, rot=R, num_voxels=nv, atom_types=['S'], vdw=1.80)
    return np.stack([c1, c2, c3, c4], axis=0)


def get_8channel_voxels_around_res(atoms, ca, cb, res, R, nv=20):
    atoms_r = [a for a in atoms if a.res == res]
    atoms_a = [a for a in atoms if a.chain_id in ca]
    atoms_b = [a for a in atoms if a.chain_id in cb]
    mkAB = get_3d_voxels_around_res(atoms, res, ones, rot=R, num_voxels=nv, atom_types=CNOS)
    mkA = get_3d_voxels_around_res(atoms_a, res, ones, rot=R, num_voxels=nv, atom_types=CNOS)
    mkB = get_3d_voxels_around_res(atoms_b, res, ones, rot=R, num_voxels=nv, atom_types=CNOS)
    mkR = get_3d_voxels_around_res(atoms_r, res, ones, rot=R, num_voxels=nv, atom_types=CNOS)
    ch1 = get_3d_voxels_around_res(atoms, res, ones, rot=R, num_voxels=nv, atom_types=['C'])
    ch2 = get_3d_voxels_around_res(atoms, res, ones, rot=R, num_voxels=nv, atom_types=['N'])
    ch3 = get_3d_voxels_around_res(atoms, res, ones, rot=R, num_voxels=nv, atom_types=['O'])
    ch4 = get_3d_voxels_around_res(atoms, res, ones, rot=R, num_voxels=nv, atom_types=['S'])
    return np.stack([mkR, mkAB, mkA, mkB, ch1, ch2, ch3, ch4], axis=0)


# def hydrophobicity(atoms, atom_types):
#     return np.asarray([ARGP820101[a.res.name] for a in atoms if a.type in atom_types])
#
#
# def molweight(atoms, atom_types):
#     return np.asarray([FASG760101[a.res.name] for a in atoms if a.type in atom_types])
#
#
# def bfactor(atoms, atom_types):
#     return np.asarray([a.temp for a in atoms if a.type in atom_types])
#
#
# def get_feature_maps(atoms, res, R, nv=20):
#     ch = get_3d_voxels_around_res(atoms, res, hydrophobicity, rot=R, num_voxels=nv, atom_types=['C', 'N', 'O', 'S'])
#     cm = get_3d_voxels_around_res(atoms, res, molweight, rot=R, num_voxels=nv, atom_types=['C', 'N', 'O', 'S'])
#     cb = get_3d_voxels_around_res(atoms, res, bfactor, rot=R, num_voxels=nv, atom_types=['C', 'N', 'O', 'S'])
#     return np.stack([ch, cm, cb], axis=0)
#
#
# def get_voxels(atoms, res, R, nv=20):
#     ca = get_4channel_voxels(atoms, res, R, nv)
#     cf = get_feature_maps(atoms, res, R, nv)
#     return ca, cf


def select_atoms_in_box(atoms, center, r):
    X = np.asarray([a.coord for a in atoms]) - center
    indx_x = np.intersect1d(np.where(X[:, 0] >= -r), np.where(X[:, 0] <= r))
    indx_y = np.intersect1d(np.where(X[:, 1] >= -r), np.where(X[:, 1] <= r))
    indx_z = np.intersect1d(np.where(X[:, 2] >= -r), np.where(X[:, 2] <= r))
    indx = reduce(np.intersect1d, [indx_x, indx_y, indx_z])
    return np.asarray(atoms)[indx]


def select_atoms_in_sphere(atoms, center, r):
    X = np.asarray([a.coord for a in atoms]) - center
    indx = np.where(X[:, 0] ** 2 + X[:, 1] ** 2 + X[:, 2] ** 2 <= r ** 2)
    return np.asarray(atoms)[indx]


def select_atoms_in_shell(center, atoms, inner, outer):
    assert outer > inner
    X = np.asarray([a.coord for a in atoms]) - center
    indx_inn = np.where(X[:, 0] ** 2 + X[:, 1] ** 2 + X[:, 2] ** 2 >= inner ** 2)
    indx_out = np.where(X[:, 0] ** 2 + X[:, 1] ** 2 + X[:, 2] ** 2 <= outer ** 2)
    indx = reduce(np.intersect1d, [indx_inn, indx_out])
    return np.asarray(atoms)[indx]


def get_descriptors_in_shell(atoms, res, descriptor_from_residues, inner, outer):
    X = np.asarray([a.coord for a in atoms]) - res.ca.coord
    indx_in = np.where(X[:, 0] ** 2 + X[:, 1] ** 2 + X[:, 2] ** 2 >= inner ** 2)
    indx_out = np.where(X[:, 0] ** 2 + X[:, 1] ** 2 + X[:, 2] ** 2 <= outer ** 2)
    indx = reduce(np.intersect1d, [indx_in, indx_out])
    residues = set([a.res for a in atoms[indx]])
    return descriptor_from_residues(res, residues)


def get_atoms_in_sphere_around_center(center, atoms, rad, atom_types=CNOS):
    return select_atoms_in_sphere([a for a in atoms if a.type in atom_types], center, rad)


def get_atoms_in_sphere_around_res(res, atoms, rad, ignore_list=BACKBONE_ATOMS):
    neighbors = set()
    atoms = select_atoms_in_sphere(atoms, res.ca.coord, MAX_DISTANCE_CUTOFF)    # save time heuristic
    for c in [a for a in res.atoms if a.name not in ignore_list]:
        hits = get_atoms_in_sphere_around_center(c.coord, atoms, rad, atom_types=CNOS)
        neighbors.update(hits)
    return list(neighbors)


def get_atoms_in_shell_around_center(center, atoms, inner, outer, atom_types=CNOS):
    return select_atoms_in_shell(center, [a for a in atoms if a.type in atom_types], inner, outer)


def get_atoms_in_shell_around_res(res, atoms, inner, outer, ignore_list=BACKBONE_ATOMS):
    neighbors = set()
    atoms = select_atoms_in_sphere(atoms, res.ca.coord, MAX_DISTANCE_CUTOFF)    # save time heuristic
    for c in [a for a in res.atoms if a.name not in ignore_list]:
        hits = get_atoms_in_shell_around_center(c.coord, atoms, inner, outer, atom_types=CNOS)
        neighbors.update(hits)
    return list(neighbors)


def get_cp_in_shell_around_res(center_res, atoms, inner, outer, M=BASU010101):
    i, w, A = center_res.index, center_res.name, center_res.chain.chain_id
    neighbors = get_atoms_in_sphere_around_center(center_res.ca.coord, atoms, inner, outer)
    residues = set([a.res for a in neighbors])
    indices_A = [(res.index, res.name) for res in residues if res.chain.chain_id == A and res.index != i]
    indices_B = [(res.index, res.name) for res in residues if res.chain.chain_id != A]
    cp_A = dict([(m, sum([M[(r, m)] - M[(r, w)] for j, r in indices_A])) for m in amino_acids])
    cp_B = dict([(m, sum([M[(r, m)] - M[(r, w)] for j, r in indices_B])) for m in amino_acids])
    return cp_A, cp_B


def get_cp_in_sphere_around_res(center_res, atoms, rad, M=BASU010101):
    i, w, A = center_res.index, center_res.name, center_res.chain.chain_id
    neighbors = get_atoms_in_sphere_around_center(center_res.ca.coord, atoms, rad)
    residues = set([a.res for a in neighbors])
    indices_A = [(res.index, res.name) for res in residues if res.chain.chain_id == A and res.index != i]
    indices_B = [(res.index, res.name) for res in residues if res.chain.chain_id != A]
    cp_A = dict([(m, sum([M[(r, m)] - M[(r, w)] for j, r in indices_A])) for m in amino_acids])
    cp_B = dict([(m, sum([M[(r, m)] - M[(r, w)] for j, r in indices_B])) for m in amino_acids])
    return cp_A, cp_B


def get_hse_in_sphere(center_res, atoms, rad, atom_types=["C"]):
    if center_res.cb is None or center_res.ca is None:
        return [], []
    alpha = center_res.ca.coord
    atoms = np.asarray([a for a in atoms if a.type in atom_types])
    X = np.asarray([a.coord for a in atoms]) - alpha
    indx = np.where(X[:, 0] ** 2 + X[:, 1] ** 2 + X[:, 2] ** 2 <= rad ** 2)
    beta = np.asarray(center_res.cb.coord) - alpha
    indx_up = np.where(np.dot(X, beta) / np.sqrt(sum(beta**2)) > 0)
    indx_down = np.where(np.dot(X, beta) / np.sqrt(sum(beta**2)) < 0)
    indx_up = reduce(np.intersect1d, [indx, indx_up])
    indx_down = reduce(np.intersect1d, [indx, indx_down])
    aas_up = set([a.res for a in atoms[indx_up]]) - {center_res}
    aas_down = set([a.res for a in atoms[indx_down]]) - {center_res}
    return aas_up, aas_down


def get_cp_descriptors_around_res(atoms, res):
    cp46A, cp46B = get_cp_in_shell_around_res(res, atoms, 4.0, 6.0)
    cp68A, cp68B = get_cp_in_shell_around_res(res, atoms, 6.0, 8.0)
    cp46 = [cp46A[m] + cp46B[m] for m in amino_acids]
    cp68 = [cp68A[m] + cp68B[m] for m in amino_acids]
    return np.concatenate([cp46, cp68], axis=0)


def get_3d_voxels_around_res(atoms, res, values_from_atoms, rot=None, num_voxels=20, atom_types=['C'], vdw=0.0):
    smooth = vdw / 3.0
    r = num_voxels // 2
    X = np.asarray([a.coord for a in atoms if a.type in atom_types])
    y = values_from_atoms(atoms, atom_types)
    if len(X) == 0:
        return voxelize([], [], num_voxels, smooth=smooth)
    X -= res.ca.coord
    if rot is not None:
        X = rot(X)
    indx_x = np.intersect1d(np.where(X[:, 0] > -r), np.where(X[:, 0] < r))
    indx_y = np.intersect1d(np.where(X[:, 1] > -r), np.where(X[:, 1] < r))
    indx_z = np.intersect1d(np.where(X[:, 2] > -r), np.where(X[:, 2] < r))
    indx = reduce(np.intersect1d, [indx_x, indx_y, indx_z])
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


def get_xyz_rotations(circle_frac=0.25):
    rotations_x = [rot_x(r * 2 * math.pi) for r in np.arange(0, .99, circle_frac)]
    rotations_y = [rot_y(r * 2 * math.pi) for r in np.arange(0, .99, circle_frac)]
    rotations_z = [rot_z(r * 2 * math.pi) for r in np.arange(0, .99, circle_frac)]
    return rotations_x + rotations_y + rotations_z


def voxelize(X, y, num_voxels, smooth=0.0):
    n = num_voxels
    voxels = np.zeros((n + 1, n + 1, n + 1))
    for coord, val in zip(X, y):
        i, j, k = np.floor(coord + n / 2)
        voxels[int(i), int(j), int(k)] = val
    if smooth != 0.0:
        voxels[:, :, :] = gaussian_filter(voxels, sigma=smooth, order=0, output=None, mode='reflect', cval=0.0, truncate=4.0)
    return voxels


if __name__ == "__main__":
    pass