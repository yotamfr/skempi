import zipfile
import numpy as np

from stride import *
from skempi_lib import *
from torch_utils import *

np.seterr("raise")


EPS = 1e-6
REPO_PATH = osp.join('..', '3DCNN_data', 'pdbs.zip')
TRAINING_SET_PATH = osp.join('..', '3DCNN_data', "train_pdbs")
VALIDATION_SET_PATH = osp.join('..', '3DCNN_data', "valid_pdbs")

PDB_ZIP = zipfile.ZipFile(REPO_PATH)
MANIFEST = [f.filename for f in PDB_ZIP.infolist()]

TRAINING_SET = [l.strip() for l in open(TRAINING_SET_PATH, "r").readlines()]
VALIDATION_SET = [l.strip() for l in open(VALIDATION_SET_PATH, "r").readlines()]
DEBUG_SET = ["1a4y", "1cse", "1kms", "1ddn", "4nos"]


def dot(u, v):
    return torch.bmm(u.unsqueeze(1), v.unsqueeze(2)).view(-1, 1)


def normalize(v):
    return (v / v.norm(p=2, dim=1).unsqueeze(1)).unsqueeze(1)


def ref_mat(ca, c, n):
    v1, v2 = ca - c, ca - n
    v3 = v2-(dot(v1, v2)/dot(v1, v1))*v1  # graham-schmidt
    v4 = torch.cross(v1, v3, dim=1)   # right hand rule
    A = torch.cat([normalize(v1), normalize(v3), normalize(v4)], 1)  # A is isometric
    # print(np.linalg.det(A.data.cpu().numpy()))   # det(A) == 1
    return A


def voxelize(xyz, channel, temp, rad, reso, num_channels=24, compute_distance_from_voxel_centers=True):
    n = int(2.0 * rad / reso) + 1
    voxels = torch.zeros((num_channels, n, n, n), device=device)
    shifted_xyz = xyz / reso + n / 2.0
    indices = shifted_xyz.long()
    if indices.size()[0] == 0:
        return voxels
    if compute_distance_from_voxel_centers:
        voxel_centers = indices.float() + reso / 2.0
        distances = (shifted_xyz - voxel_centers).norm(p=2, dim=1)
        distances = reso - distances
        voxels[channel[:, 0], indices[:, 0], indices[:, 1], indices[:, 2]] += distances
        voxels[channel[:, 1], indices[:, 0], indices[:, 1], indices[:, 2]] += distances
    else:
        voxels[channel[:, 0], indices[:, 0], indices[:, 1], indices[:, 2]] += 1
        voxels[channel[:, 1], indices[:, 0], indices[:, 1], indices[:, 2]] += 1
    # voxels[num_channels-1, indices[:, 0], indices[:, 1], indices[:, 2]] += temp.squeeze(1)
    return voxels


def get_channels(atom):
    return [CNOS.index(atom.type), len(CNOS)+amino_acids.index(atom.res.name)]


def _preprocess_atoms_and_residues(atoms, residues, rad, epsilon=0.0):
    ca = torch.tensor([res.ca.coord for res in residues], dtype=torch.float, device=device)
    c = torch.tensor([res.c.coord for res in residues], dtype=torch.float, device=device)
    n = torch.tensor([res.n.coord for res in residues], dtype=torch.float, device=device)
    R, t = ref_mat(ca, c, n), ca.unsqueeze(1)

    X0 = [torch.tensor([atm.coord for atm in filter_atoms(res.atoms)],
                       dtype=torch.float, device=device) for res in residues]
    T0 = [torch.tensor([get_channels(atm) for atm in filter_atoms(res.atoms)],
                       dtype=torch.long, device=device) for res in residues]
    Y0 = [torch.tensor([[np.log(max(atm.temp, EPS))] for atm in filter_atoms(res.atoms)],
                       dtype=torch.float, device=device) for res in residues]

    X = torch.tensor([atm.coord for atm in atoms],
                     dtype=torch.float, device=device).unsqueeze(0).repeat(len(residues), 1, 1)
    T = torch.tensor([get_channels(atm) for atm in atoms],
                     dtype=torch.long, device=device).unsqueeze(0).repeat(len(residues), 1, 1)
    Y = torch.tensor([[np.log(max(atm.temp, EPS))] for atm in atoms],
                     dtype=torch.float, device=device).unsqueeze(0).repeat(len(residues), 1, 1)

    X = torch.bmm(R, (X - t).transpose(1, 2)).transpose(1, 2)
    if epsilon > 0: X.add(torch.randn_like(X)*epsilon)

    ixx = (X[:, :, 0] <= rad) & (X[:, :, 0] >= -rad)
    ixy = (X[:, :, 1] <= rad) & (X[:, :, 1] >= -rad)
    ixz = (X[:, :, 2] <= rad) & (X[:, :, 2] >= -rad)
    ix = (ixx & ixy & ixz)

    return X, X0, Y, Y0, T, T0, R, t, ix


def filter_residues(residues):
    return [res for res in residues if res.ca and res.c and res.n and (res.name in amino_acids)]


def filter_atoms(atoms):
    # return [atm for atm in atoms if (atm.type in CNOS) and (atm.res.name in amino_acids) and (atm.temp > 0)]
    return [atm for atm in atoms if (atm.type in CNOS) and (atm.res.name in amino_acids)]


def load_single_position(st, res, rad=16., reso=1.25):
    atoms, residues = filter_atoms(st.atoms), filter_residues(st.residues)
    X, X0, Y, Y0, T, T0, R, t, ix = _preprocess_atoms_and_residues(atoms, [res], rad)
    resi = 0
    x0 = torch.mm(R[resi, :, :], (X0[resi] - t[resi]).transpose(0, 1)).transpose(0, 1)
    y = voxelize(X[resi, ix[resi], :], T[resi, ix[resi], :], Y[resi, ix[resi], :], rad, reso)
    x = voxelize(x0, T0[resi], Y0[resi], rad, reso)
    aa = amino_acids.index(res.name)
    return aa, (y - x), y


def pdb_loader(repo, list_of_pdbs, n_iter, rad=16., reso=1.25, sample_ratio=0.1, handle_error=None):
    i_iter, i_pdb = 0, 0
    p = np.random.permutation(len(list_of_pdbs))
    while i_iter < n_iter:
        pdb = list_of_pdbs[p[i_pdb]]
        i_pdb = (i_pdb + 1) % len(list_of_pdbs)
        try:
            st = parse_pdb(pdb, StringIO(repo.read("pdbs/%s.pdb" % pdb)))
        except KeyError:
            continue
        atoms, residues = filter_atoms(st.atoms), filter_residues(st.residues)
        if len(residues) == 0:
            continue
        m = int(len(residues) * sample_ratio)
        residues = np.random.choice(residues, m, replace=False)

        try:
            X, X0, Y, Y0, T, T0, R, t, ix = _preprocess_atoms_and_residues(atoms, residues, rad)
        except RuntimeError as e:
            if handle_error:
                handle_error(pdb, e)
            continue

        for resi, res in enumerate(residues):
            if i_iter == n_iter:
                break
            try:
                x0 = torch.mm(R[resi, :, :], (X0[resi] - t[resi]).transpose(0, 1)).transpose(0, 1)
                y = voxelize(X[resi, ix[resi], :], T[resi, ix[resi], :], Y[resi, ix[resi], :], rad, reso)
                x = voxelize(x0, T0[resi], Y0[resi], rad, reso)
                aa = amino_acids.index(res.name)
            except RuntimeError as e:
                if handle_error:
                    handle_error(pdb, e)
                continue

            i_iter += 1
            yield aa, y - x, y


def batch_generator(loader, batch_size=32):
    def prepare_batch(data):
        a, x, y = zip(*data)
        x = torch.stack(x, 0)
        y = torch.stack(y, 0)
        a = torch.tensor(a, dtype=torch.long, device=device)
        return a, x, y

    batch = []
    for inp in loader:
        if len(batch) == batch_size:
            yield prepare_batch(batch)
            batch = []
        batch.append(inp)
    yield prepare_batch(batch)


if __name__ == "__main__":
    for x, y in batch_generator(pdb_loader(PDB_ZIP, TRAINING_SET, 1000)):
        assert x.size() == y.size()
