import time
import zipfile
import numpy as np
import os.path as osp
from tqdm import tqdm
from StringIO import StringIO
from concurrent.futures import *
from collections import deque

from grid_utils import *

np.seterr("raise")

REPO_PATH = osp.join('..', '3DCNN_data', 'pdbs.zip')
TRAINING_SET_PATH = osp.join('..', '3DCNN_data', "train_pdbs")
VALIDATION_SET_PATH = osp.join('..', '3DCNN_data', "valid_pdbs")

PDB_ZIP = zipfile.ZipFile(REPO_PATH)
MANIFEST = [f.filename for f in PDB_ZIP.infolist()]

TRAINING_SET = [l.strip() for l in open(TRAINING_SET_PATH, "r").readlines()]
VALIDATION_SET = [l.strip() for l in open(VALIDATION_SET_PATH, "r").readlines()]
DEBUG_SET = ["1a4y", "1cse", "1kms", "1ddn", "4nos"]


class PdbLoader(object):

    def __init__(self, reader, num_iterations, num_augmentations=1):
        self.reader = reader
        self.num_iter = num_iterations
        self.num_aug = num_augmentations
        self.curr = 0
        self.load()

    def reset(self):
        self.curr = 0
        self.load()

    def load(self):
        i = 0
        pbar = tqdm(range(self.num_iter), desc="loading data...")
        while i < self.num_iter:
            try:
                self.reader.read()
                pbar.update(1)
                i += 1
            except StopIteration:
                self.reader.reset()
        pbar.close()

    def next(self):
        if self.curr < len(self):
            ret = None
            total_time_slept = 0.0
            while ret is None:
                if total_time_slept > 60:
                    raise StopIteration
                try:
                    ret = self.reader.Q.pop()
                except IndexError:
                    time.sleep(0.1)
                    total_time_slept += 0.1
            self.curr += 1
            return ret
        else:
            raise StopIteration

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self

    def __len__(self):
        return self.num_iter * self.num_aug


class PdbReader(object):
    def __init__(self, producer, list_of_pdbs, rotations=[None], repo=PDB_ZIP, step=10, num_voxels=20):
        self._list_of_pdbs = list_of_pdbs
        self._p = np.random.permutation(len(list_of_pdbs))
        self._step = step
        self._repo = repo
        self.nv = num_voxels
        self.rotations = rotations
        self.func = producer
        self.Q = deque()
        self.E = ThreadPoolExecutor(1)
        self.reset()

    def reset(self):
        self._pdb_ix = -1
        self.struct = None
        self.load_next_struct()

    def read_pdb(self, pdb):
        return parse_pdb(pdb, StringIO(self._repo.read("pdbs/%s.pdb" % pdb)))

    def load_next_struct(self):
        self._pdb_ix += 1
        if self._pdb_ix == len(self._p):
            raise StopIteration
        try:
            self.struct = self.read_pdb(self.curr_pdb)
            self._chain_ix = 0
            self._res_ix = 0
        except (KeyError, NotImplementedError):   # key errors are thrown by read_pdb
            self.load_next_struct()

    @property
    def curr_pdb(self):
        return self._list_of_pdbs[self._p[self._pdb_ix]]

    @property
    def curr_chain(self):
        return self.struct._chains[self._chain_ix]

    def read(self):

        if self._res_ix + self._step < len(self.curr_chain):
            self._res_ix += self._step
        elif self._chain_ix + 1 < len(self.struct._chains):
            self._chain_ix += 1
            self._res_ix = 0
        else:
            self.load_next_struct()
            return self.read()

        struct = self.struct
        res = self.curr_chain[self._res_ix]

        if res.ca is None:
            return self.read()

        self.E.submit(self.func, self.Q, struct, res, self.rotations, self.nv)


def non_blocking_producer_v1(queue, struct, res, rotations, nv=20):
    atoms = select_atoms_in_sphere(struct.atoms, res.ca.coord, nv)
    ix = list(amino_acids).index(res.name)
    descriptors = [int(i == ix) for i in range(len(amino_acids))]
    for rot in rotations:
        voxels = get_4channel_voxels_around_res(atoms, res, rot, nv=nv)
        queue.appendleft([voxels, descriptors])


def non_blocking_producer_v2(queue, struct, res, rotations, nv=20):
    atoms = select_atoms_in_sphere(struct.atoms, res.ca.coord, nv)
    descriptors = get_cp_descriptors_around_res(atoms, res)
    for rot in rotations:
        voxels = get_4channel_voxels_around_res(atoms, res, rot, nv=nv)
        queue.appendleft([voxels, descriptors])


def non_blocking_producer_v3(queue, struct, res, rotations, nv=20):
    atoms = select_atoms_in_sphere(struct.atoms, res.ca.coord, nv)
    descriptors1 = get_cp_descriptors_around_res(atoms, res)
    ix = list(amino_acids).index(res.name)
    descriptors2 = [int(i == ix) for i in range(len(amino_acids))]
    for rot in rotations:
        voxels = get_4channel_voxels_around_res(atoms, res, rot, nv=nv)
        descriptors = np.concatenate([descriptors2, descriptors1], axis=0)
        queue.appendleft([voxels, descriptors])


if __name__ == "__main__":
    rotations = get_xyz_rotations(.25)
    reader = PdbReader(non_blocking_producer_v2, TRAINING_SET, rotations, step=10, num_voxels=20)
    loader = PdbLoader(reader, 50000, len(rotations))
    pbar = tqdm(range(len(loader)), desc="processing data...")
    for _, (inp, tgt) in enumerate(loader):
        pbar.update(1)
        msg = "qsize: %d" % (len(loader.reader.Q),)
        assert inp.shape == (4, 21, 21, 21)
        assert tgt.shape == (40,)
        pbar.set_description(msg)
