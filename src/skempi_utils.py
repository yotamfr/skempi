import sys
import pickle

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

import numpy as np
import pandas as pd
import os.path as osp
from tqdm import tqdm

from sklearn.metrics.pairwise import euclidean_distances
from pymongo import MongoClient

mongo_url = "mongodb://localhost:27017/"
db_name = "prot2vec"
client = MongoClient(mongo_url)
db = client[db_name]
collection = db.skempi_uniprot20


try:
    from src.pdb_utils import *
    from src.skempi_consts import *
except ImportError:
    from pdb_utils import *
    from skempi_consts import *


def get_distance_matrix(atoms1, atoms2):
    X = np.matrix([a.coord for a in atoms1])
    Y = np.matrix([a.coord for a in atoms2])
    return euclidean_distances(X, Y)


class SkempiChain(object):
    def __init__(self, structure, chain):
        self.pdb = structure.id
        self.chain = self.struct[chain]

    @property
    def seq(self):
        return "".join([res.name for res in self.residues])

    @property
    def id(self):
        return "%s_%s" % (self.pdb, self.chain)


class Mutation(object):

    def __init__(self, mutation):
        try:
            self.w = mutation[0]
            self.chain_id = mutation[1]
            self.i = int(mutation[2:-1]) - 1
            self.m = mutation[-1]
            self.ins_code = None

        except ValueError:
            self.w = mutation[0]
            self.chain_id = mutation[1]
            self.i = int(mutation[2:-2]) - 1
            self.m = mutation[-1]
            self.ins_code = mutation[-2]

    def __str__(self):
        return str(vars(self))


class MSA(object):
    def __init__(self, pdb, chain):
        self._msa = collection.find_one({
            "_id": "%s_%s" % (pdb, chain)})["alignment"]

    def __getitem__(self, i):
        return self._msa[i]


class Profile(object):
    def __init__(self, pdb, chain):
        self._profile = collection.find_one({
            "_id": "%s_%s" % (pdb, chain)})["profile"]

    def __getitem__(self, t):
        i, a = t
        return self._profile[i][a]


class Stride(object):
    def __init__(self, pdb):
        df = pd.read_csv('../data/stride/%s.out' % pdb)
        self._dict = {}
        for i, row in df.iterrows():
            self._dict[(row.Chain, row.Res - 1)] = row.to_dict()

    def __getitem__(self, t):
        chain, res = t
        return self._dict[(chain, res)]


class SkempiRecord(object):
    def __init__(self, pdb, chains_a, chains_b):
        fd = open(osp.join(PDB_PATH, "%s.pdb" % pdb), 'r')
        self.struct = parse_pdb(pdb, fd)
        self.pdb = pdb
        self.chains_a = {c: self.struct[c] for c in chains_a}
        self.chains_b = {c: self.struct[c] for c in chains_b}
        self.res_chain_to_atom_indices = {}
        self.atom_indices_to_chain_res = {}
        self.atoms = []
        self.init_dictionaries()
        self.dist_mat = None
        self._profiles = {}
        self.init_profiles()
        self._stride = Stride(self.pdb)

    @property
    def chains(self):
        return self.struct.chains

    def init_profiles(self):
        self._profiles = {c: Profile(self.pdb, c) for c in self.chains}

    def get_profile(self, chain_id):
        return self._profiles[chain_id]

    @property
    def stride(self):
        return self._stride

    def init_dictionaries(self):
        for chain in self.struct:
            for res_i, res in enumerate(chain):
                for atom in res:
                    chain_id = chain.chain_id
                    if (chain_id, res_i) in self.res_chain_to_atom_indices:
                        self.res_chain_to_atom_indices[(chain_id, res_i)].append(len(self.atoms))
                    else:
                        self.res_chain_to_atom_indices[(chain_id, res_i)] = [len(self.atoms)]
                    self.atom_indices_to_chain_res[len(self.atoms)] = (chain_id, res_i)
                    self.atoms.append(atom)

    def compute_dist_mat(self):
        atoms = self.atoms
        self.dist_mat = get_distance_matrix(atoms1=atoms, atoms2=atoms)

    def __getitem__(self, chain):
        return self.chains[chain]

    def get_sphere_indices(self, chain, res_i, threshold):
        mat = self.dist_mat
        row_indices = self.res_chain_to_atom_indices[(chain, res_i)]
        col_indices = []
        for row_i in row_indices:
            col_indices.extend([ix for ix in np.where(mat[row_i] <= threshold)[0]])
        return set([self.atom_indices_to_chain_res[col_i] for col_i in col_indices])

    def get_stride(self):
        pass

    def __iter__(self):
        for chain in self.chains.values():
            yield chain

    def to_fasta(self):
        pass


def to_fasta(skempi_entries, out_file):
    sequences = []
    for entry in skempi_entries:
        for chain in entry:
            print(chain, len(chain))
            sequences.append(SeqRecord(Seq(chain.seq), chain.id))
    SeqIO.write(sequences, open(out_file, 'w+'), "fasta")


def EI(m, w, P, i, B):
    return sum([P[(i, a)] * (B[(a, m)] - B[(a, w)]) for a in amino_acids])


def save_object(obj, filename):
    with open(filename, 'w+b') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def load_object(pth):
    with open(pth, 'rb') as f:
        loaded_dist_mat = pickle.load(f)
        assert len(loaded_dist_mat) > 0
    return loaded_dist_mat


if __name__ == "__main__":

    skempi_df = pd.read_excel(osp.join('../data', 'SKEMPI_1.1.xlsx'))

    prots = skempi_df.Protein.values
    skempi_entries = []

    for t in tqdm(set([tuple(pdb_str.split('_')) for pdb_str in prots]),
                  desc="skempi entries processed"):
        skempi_entries.append(SkempiRecord(*t))

    to_fasta(skempi_entries, "../data/skempi.fas")
