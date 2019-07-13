import math
import numpy as np
from pyobb.obb import OBB
from functools import reduce
from scipy.linalg import expm, norm
from scipy.ndimage.filters import gaussian_filter
from scipy.linalg import get_blas_funcs

from pdb_utils import *
from pytorch_utils import *
from aaindex import *

VDW = {'C': 1.7, 'O': 1.52, 'N': 1.55, 'S': 1.8}

CNOS = "CNOS"

CHET = "CHET"

MAX_DISTANCE_CUTOFF = 15.0

default = None


def get_center_and_rotation(residues):
    if len(residues) == 1:
        res = residues[0]
        c = np.asarray(res.c.coord) - res.center
        n = np.asarray(res.n.coord) - res.center
        ca = np.asarray(res.ca.coord) - res.center
        return res.center, ref_mat(ca, c, n)
    else:
        points = [res.center for res in residues]
        obb = OBB.build_from_points(points)
        return obb.centroid, obb.rotation


def get_pdb_voxel_channels(atoms, center, rot, rad=10.0, reso=1.0):
    atoms, X = select_atoms_in_box(atoms, center, rot, rad)
    temp = get_voxel_channel(atoms, X, lambda x: True, lambda x: x.temp, rad=rad, reso=reso)
    atm_channels = [get_voxel_channel(atoms, X, lambda x: x.type == t, lambda x: default, rad=rad, reso=reso) for t in 'CNOS']
    res_channels = [get_voxel_channel(atoms, X, lambda x: x.res.name == aa, lambda x: default, rad=rad, reso=reso) for aa in amino_acids]
    ss_channels = [get_voxel_channel(atoms, X, lambda x: x.res.ss == t, lambda x: default, rad=rad, reso=reso) for t in 'CHET']
    return np.stack(atm_channels + [temp] + res_channels + ss_channels, axis=0)


def get_skempi_voxel_channels(atoms, center, rot, rad=10.0, reso=1.0):
    atoms, X = select_atoms_in_box(atoms, center, rot, rad)
    atm_channels = [get_voxel_channel(atoms, X, lambda x: x.type == t, lambda x: default, rad=rad, reso=reso) for t in 'CNOS']
    # res_channels = [get_voxel_channel(atoms, X, lambda x: x.res.name == aa, lambda x: default, rad=rad, reso=reso) for aa in amino_acids]
    c5 = get_voxel_channel(atoms, X, lambda x: True, lambda x: x.res.consv, rad=rad, reso=reso)
    # c6 = get_voxel_channel(atoms, X, lambda x: True, lambda x: x.res.acc1, rad=rad, reso=reso)
    # c7 = get_voxel_channel(atoms, X, lambda x: True, lambda x: x.res.acc2, rad=rad, reso=reso)
    c8 = get_voxel_channel(atoms, X, lambda x: True, lambda x: x.temp, rad=rad, reso=reso)
    # return np.stack(atm_channels + [c5, c6, c7, c8], axis=0)
    return np.stack(atm_channels + [c5, c8], axis=0)


def get_voxel_channels_of_residues(atoms, residues, center, rot, rad=10.0, reso=1.0):
    atoms, X = select_atoms_in_box(atoms, center, rot, rad)
    atm_channels = [get_voxel_channel(atoms, X, lambda x: (x.type == t) and (x.res in residues), lambda x: default, rad=rad, reso=reso) for t in 'CNOS']
    res_channels = [get_voxel_channel(atoms, X, lambda x: (x.res.name == aa) and (x.res in residues), lambda x: default, rad=rad, reso=reso) for aa in amino_acids]
    return np.stack(atm_channels + res_channels, axis=0)


def fast_matrix_multiplication(A, B):
    return get_blas_funcs("gemm", [A, B])(1, A, B)


def select_atoms_in_box(atoms, center, R, r):
    X = np.asarray([a.coord for a in atoms]) - center
    X = get_blas_funcs("gemm", [R, X.T])(1, R, X.T).T   # fast matrix multiplication using BLAS
    indx_x = np.intersect1d(np.where(X[:, 0] >= -r), np.where(X[:, 0] <= r))
    indx_y = np.intersect1d(np.where(X[:, 1] >= -r), np.where(X[:, 1] <= r))
    indx_z = np.intersect1d(np.where(X[:, 2] >= -r), np.where(X[:, 2] <= r))
    indx = reduce(np.intersect1d, [indx_x, indx_y, indx_z])
    return np.asarray(atoms)[indx], X[indx]


def select_atoms_in_box_cuda(atoms, center, rot, r):
    X = torch.tensor([a.coord for a in atoms], dtype=torch.float, device=device)
    c = torch.tensor(center, dtype=torch.float, device=device)
    R = torch.tensor(rot, dtype=torch.float, device=device)
    X = torch.mm(R, (X-c).transpose(0, 1)).transpose(0, 1)
    ixx = (X[:, 0] >= -r) & (X[:, 0] <= r)
    ixy = (X[:, 1] >= -r) & (X[:, 1] <= r)
    ixz = (X[:, 2] >= -r) & (X[:, 2] <= r)
    ix = ixx & ixy & ixz
    return np.asarray(atoms)[ix.data.cpu().numpy().astype(bool)], X[ix].data.cpu().numpy()


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


def get_atoms_in_sphere_around_res(res, atoms, rad):
    neighbors = set()
    atoms = select_atoms_in_sphere(atoms, res.center, MAX_DISTANCE_CUTOFF)    # save time heuristic
    for a in res.atoms:
        hits = get_atoms_in_sphere_around_center(a.coord, atoms, rad, atom_types=CNOS)
        hs = [h for h in hits if (h.res != res)]
        neighbors.update(hs)
    return list(neighbors)


def get_atoms_in_shell_around_center(center, atoms, inner, outer, atom_types=CNOS):
    return select_atoms_in_shell(center, [a for a in atoms if a.type in atom_types], inner, outer)


def get_atoms_in_shell_around_res(res, atoms, inner, outer):
    neighbors = set()
    atoms = select_atoms_in_sphere(atoms, res.ca.coord, MAX_DISTANCE_CUTOFF)    # save time heuristic
    for a in res.atoms:
        hits = get_atoms_in_shell_around_center(a.coord, atoms, inner, outer, atom_types=CNOS)
        hs = [h for h in hits if (h.res != res)]
        neighbors.update(hs)
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


def get_voxel_channel(atoms, X, selector_fn, value_fn, rad=10.0, reso=1.0, vdw=0.0):
    smooth = vdw / 3.0
    try:
        indx = np.vectorize(selector_fn)(atoms)
        y = np.vectorize(value_fn)(atoms[indx])
        return voxelize(X[indx], y, rad, reso, smooth=smooth)
    except ValueError:
        return voxelize([], [], rad, reso, smooth=smooth)


def ref_mat(ca, c, n):
    v1, v2 = ca - c, ca - n
    v3 = v2-(np.dot(v1, v2)/np.dot(v1, v1))*v1  # graham-schmidt
    v4 = np.cross(v1, v3)   # right hand rule
    v1 /= np.linalg.norm(v1)
    v3 /= np.linalg.norm(v3)
    v4 /= np.linalg.norm(v4)
    A = np.matrix([v1, v3, v4])  # A is isometric
    # print(np.linalg.det(A))   # det(A) == 1
    return A


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


def voxelize(X, y, radius, resolution=1.0, smooth=0.0):
    n = int(2.0 * radius / resolution)
    voxels = np.zeros((n, n, n))
    if len(X) == 0: return voxels
    shifted_X = X/resolution + n/2.0
    indices = shifted_X.astype(int)
    voxel_centers = indices + resolution/2.0
    distances = np.linalg.norm(shifted_X - voxel_centers, axis=1)
    distances = resolution - distances
    for ix, ((i, j, k), v) in enumerate(zip(indices.tolist(), y)):
        try:
            voxels[i, j, k] = v if v else distances[ix]
        except IndexError:
            continue
    if smooth != 0.0:
        voxels[:, :, :] = gaussian_filter(voxels, sigma=smooth, order=0, output=None, mode='reflect', cval=0.0, truncate=4.0)
    return voxels


def plot_3d_ss_voxels(voxels, res_voxels=None):
    from mpl_toolkits.mplot3d import axes3d, Axes3D
    import matplotlib.pyplot as plt
    import numpy as np
    C = voxels[0].astype(bool)
    H = voxels[1].astype(bool)
    E = voxels[2].astype(bool)
    T = voxels[3].astype(bool)
    B = res_voxels[0].astype(bool)
    for i in range(1, 4): B |= res_voxels[i].astype(bool)
    voxels = C | H | E | T | B
    # set the colors of each object
    colors = np.empty(voxels.shape, dtype=object)
    colors[C] = 'green'
    colors[H] = 'red'
    colors[E] = 'yellow'
    colors[T] = 'cyan'
    colors[B] = 'black'
    # and plot everything
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(voxels, facecolors=colors, edgecolor='k')
    return plt


def plot_3d_atom_voxels(voxels):
    from mpl_toolkits.mplot3d import axes3d, Axes3D
    import matplotlib.pyplot as plt
    import numpy as np
    C = voxels[0].astype(bool)
    N = voxels[1].astype(bool)
    O = voxels[2].astype(bool)
    S = voxels[3].astype(bool)
    voxels = C | N | O | S
    # set the colors of each object
    colors = np.empty(voxels.shape, dtype=object)
    colors[C] = 'green'
    colors[N] = 'blue'
    colors[O] = 'red'
    colors[S] = 'purple'
    # and plot everything
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(voxels, facecolors=colors, edgecolor='k')
    return plt


def plot_3d_residue_voxels(voxels):
    from mpl_toolkits.mplot3d import axes3d, Axes3D
    import matplotlib.pyplot as plt
    import numpy as np
    AA = [voxels[i].astype(bool) for i in range(20)]
    voxels = AA[0]
    for i in range(1, 20): voxels |= AA[i]
    # set the colors of each object
    colors = np.empty(voxels.shape, dtype=object)

    colors[AA[amino_acids.index("G")]] = 'grey'
    colors[AA[amino_acids.index("P")]] = 'pink'

    colors[AA[amino_acids.index("M")]] = 'yellow'
    colors[AA[amino_acids.index("A")]] = 'yellow'
    colors[AA[amino_acids.index("I")]] = 'yellow'
    colors[AA[amino_acids.index("L")]] = 'yellow'
    colors[AA[amino_acids.index("V")]] = 'yellow'

    colors[AA[amino_acids.index("F")]] = 'green'
    colors[AA[amino_acids.index("W")]] = 'green'
    colors[AA[amino_acids.index("Y")]] = 'green'

    colors[AA[amino_acids.index("N")]] = 'cyan'
    colors[AA[amino_acids.index("C")]] = 'cyan'
    colors[AA[amino_acids.index("Q")]] = 'cyan'
    colors[AA[amino_acids.index("S")]] = 'cyan'
    colors[AA[amino_acids.index("T")]] = 'cyan'

    colors[AA[amino_acids.index("R")]] = 'blue'
    colors[AA[amino_acids.index("H")]] = 'blue'
    colors[AA[amino_acids.index("K")]] = 'blue'

    colors[AA[amino_acids.index("D")]] = 'red'
    colors[AA[amino_acids.index("E")]] = 'red'

    # and plot everything
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(voxels, facecolors=colors, edgecolor='k')
    return plt


if __name__ == "__main__":
    from skempi_lib import *
    from skempi_consts import *
    df = skempi_df_v2
    df, lim = skempi_df_v2, 1000
    for r in load_skempi(df[df.version == 1][:lim].reset_index(drop=True), SKMEPI2_PDBs, False):
        if len(r.mutations) > 1:
            points = [r.struct.get_residue(m).center for m in r.mutations]
            obb = OBB.build_from_points(points)
            cnt, rot = obb.centroid, obb.rotation
