import os
import sys

import numpy as np
import pandas as pd
import os.path as osp


from aaindex import *
from pdb_utils import *
from grid_utils import *


class Mutation(object):

    def __init__(self, mutation_str):
        self._str = mutation_str
        iw, im = (0, -1)
        try:
            self.w = mutation_str[iw]
            self.chain_id = mutation_str[1]
            self.i = int(mutation_str[2:-1]) - 1
            self.m = mutation_str[im]
            self.ins_code = None

        except ValueError:
            self.w = mutation_str[iw]
            self.chain_id = mutation_str[1]
            self.i = int(mutation_str[2:-2]) - 1
            self.m = mutation_str[im]
            self.ins_code = mutation_str[-2]

    def __str__(self):
        return self._str

    def __reversed__(self):
        return Mutation("%s%s%s" % (self.m, str(self)[1:-1], self.w))

    def __hash__(self):
        return hash(self._str)


def get_mutation(res, r):
    return Mutation("%s%s%d%s" % (res.name, res.chain.chain_id, res.num, r))


def get_descriptor(mutations, mat, agg=np.mean):    # MolWeight:FASG760101, Hydrophobic:ARGP820101
    return agg([mat[mut.m] - mat[mut.w] for mut in mutations])


if __name__ == "__main__":
    pass