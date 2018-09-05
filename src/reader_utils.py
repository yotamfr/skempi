import time
import zipfile
import numpy as np
import os.path as osp
from tqdm import tqdm
from StringIO import StringIO

from Queue import Queue
from concurrent.futures import *

from grid_utils import *

np.seterr("raise")

REPO_PATH = osp.join('..', '3DCNN_data', 'pdbs.zip')
TRAINING_SET_PATH = osp.join('..', '3DCNN_data', "train_pdbs")
VALIDATION_SET_PATH = osp.join('..', '3DCNN_data', "valid_pdbs")

PDB_ZIP = zipfile.ZipFile(REPO_PATH)
MANIFEST = [f.filename for f in PDB_ZIP.infolist()]

TRAINING_SET = [l.strip() for l in open(TRAINING_SET_PATH, "r").readlines()]
VALIDATION_SET = [l.strip() for l in open(VALIDATION_SET_PATH, "r").readlines()]
DEBUG_SET = ["1a4y", "1kms", "1ddn", "4nos"]

NUM_CPU = 20
E = ThreadPoolExecutor(NUM_CPU)


class PdbLoader(object):

    def __init__(self, producer, list_of_pdbs, num_iterations, rotations=[None], step=10):
        self.reader = PdbReader(producer, list_of_pdbs, rotations, step)
        self.num_iter = num_iterations
        self.num_aug = len(rotations)
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
            ret = self.reader.queue.get(block=True)
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
    def __init__(self, producer, list_of_pdbs, rotations, step=10, repo=PDB_ZIP):
        self._list_of_pdbs = list_of_pdbs
        self._p = np.random.permutation(len(list_of_pdbs))
        self._step = step
        self._repo = repo
        self.rotations = rotations
        self.func = producer
        self.queue = Queue()
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
        except (KeyError, NotImplementedError, AssertionError):
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

        res = self.curr_chain[self._res_ix]

        if res.ca is None:
            return self.read()

        E.submit(self.func, self.queue, self.struct, res, self.rotations)


def non_blocking_producer(queue, struct, res, rotations, nv=20):
    atoms = select_atoms_in_box(struct.atoms, res.ca.coord, nv)
    descriptors = get_cp_descriptors_around_res(atoms, res)
    for rot in rotations:
        voxels = get_4channel_voxels_around_res(atoms, res, rot, nv=nv)
        queue.put([voxels, descriptors])


if __name__ == "__main__":
    augmentations = get_xyz_rotations(.25)
    loader = PdbLoader(non_blocking_producer, TRAINING_SET, 1000, augmentations)
    pbar = tqdm(range(len(loader)), desc="processing data...")
    for _, (inp, tgt) in enumerate(loader):
        pbar.update(1)
        msg = "qsize: %d" % (loader.reader.queue.qsize(),)
        assert inp.shape == (4, 21, 21, 21)
        assert tgt.shape == (40,)
        pbar.set_description(msg)
