import zipfile
from pdb_utils import *
from grid_utils import *
import os.path as osp
import numpy as np
from StringIO import StringIO
from tqdm import tqdm
from concurrent.futures import *

np.seterr("raise")

REPO_PATH = osp.join('..', '3DCNN_data', 'pdbs.zip')
TRAINING_SET_PATH = osp.join('..', '3DCNN_data', "train_pdbs")
VALIDATION_SET_PATH = osp.join('..', '3DCNN_data', "valid_pdbs")

PDB_ZIP = zipfile.ZipFile(REPO_PATH)
MANIFEST = [f.filename for f in PDB_ZIP.infolist()]

TRAINING_SET = [l.strip() for l in open(TRAINING_SET_PATH, "r").readlines()]
VALIDATION_SET = [l.strip() for l in open(VALIDATION_SET_PATH, "r").readlines()]
DEBUG_SET = ["1a4y", "1kms", "1ddn", "4nos"]

NUM_CPU = 8
E = ThreadPoolExecutor(NUM_CPU)


class PdbWindowReader(object):
    def __init__(self, func, list_of_pdbs, rotations, step=10, repo=PDB_ZIP):
        self._list_of_pdbs = list_of_pdbs
        self._p = np.random.permutation(len(list_of_pdbs))
        self._step = step
        self._repo = repo
        self._rotations = rotations
        self._func = func
        self.reset()

    def reset(self):
        self._pdb_ix = -1
        self._struct = None
        self.load_next_struct()

    def read_pdb(self, pdb):
        return parse_pdb(pdb, StringIO(self._repo.read("pdbs/%s.pdb" % pdb)))

    def load_next_struct(self):
        self._pdb_ix += 1
        if self._pdb_ix == len(self._p):
            raise StopIteration
        try:
            self._struct = self.read_pdb(self.curr_pdb)
            self._chain_ix = 0
            self._res_ix = 0
            self._rot_ix = -1
        except (KeyError, NotImplementedError, AssertionError):
            self.load_next_struct()

    @property
    def curr_pdb(self):
        return self._list_of_pdbs[self._p[self._pdb_ix]]

    @property
    def curr_chain(self):
        return self._struct._chains[self._chain_ix]

    def next(self):

        if self._rot_ix + 1 < len(self._rotations):
            self._rot_ix += 1
        elif self._res_ix + self._step < len(self.curr_chain):
            self._res_ix += self._step
            self._rot_ix = 0
        elif self._chain_ix + 1 < len(self._struct._chains):
            self._chain_ix += 1
            self._res_ix = 0
            self._rot_ix = 0
        else:
            self.load_next_struct()
            return self.next()

        struct = self._struct
        res = self.curr_chain[self._res_ix]
        rot = self._rotations[self._rot_ix]

        if res.ca is None:
            self._rot_ix = len(self._rotations)
            return self.next()

        key = (struct, res, rot)
        val = E.submit(self._func, struct, res, rot)
        return key, val

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self


if __name__ == "__main__":
    reader = PdbWindowReader(get_voxels, TRAINING_SET, [None], step=10)
    for key, val in tqdm(reader, desc="processed"):
        pass