import os
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

try:
    from src.consts import *
    from src.aaindex import *
    from src.pdb_utils import *
    from src.skempi_consts import *
except ImportError:
    from consts import *
    from aaindex import *
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

    def __init__(self, mutation_str, reverse=False):
        iw, im = (-1, 0) if reverse else (0, -1)
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
        return str(vars(self))


class MSA(object):
    def __init__(self, pdb, chain):
        self._msa = collection_msa.find_one({
            "_id": "%s_%s" % (pdb, chain)})["alignment"]

    def __getitem__(self, i):
        return self._msa[i]

    def to_fasta(self, pth):
        lines = [">%s\n%s\n" % (uid if i == 0 else "SEQ%d" % i, seq)
                 for i, (uid, seq) in enumerate(self._msa)]
        with open(pth, "w+") as f:
            f.writelines(lines)


class SkempiProfile(object):
    def __init__(self, pdb, chain):
        uid = "%s_%s.prof" % (pdb, chain)
        df = pd.read_csv(osp.join("..", "data", "skempiprofiles", uid), sep=' ')
        self._profile = [df.loc[i].to_dict() for i in range(len(df))]

    def __getitem__(self, t):
        i, a = t
        pos_dict = self._profile[i]
        return pos_dict[a]  ### + 0.05 * pos_dict['-']


class Profile(object):
    def __init__(self, pdb, chain):
        uid = "%s_%s" % (pdb, chain)
        doc = collection_msa.find_one({
            "_id": uid})
        assert doc is not None
        self._profile = doc["profile"]

    def __getitem__(self, t):
        i, a = t
        return self._profile[i][a]


class Stride(object):
    def __init__(self, pdb):
        df = pd.read_csv('../data/stride/%s.out' % pdb)
        self._dict = {}
        self._total = 0.0
        for i, row in df.iterrows():
            d_row = row.to_dict()
            self._total += (d_row["ASA_Chain"] - d_row["ASA"])
            self._dict[(row.Chain, row.Res - 1)] = d_row

    def __getitem__(self, t):
        chain, res = t
        return self._dict[(chain, res)]


class SkempiRecord(object):
    def __init__(self, skempi_struct, mutations, ddg):
        self.struct = skempi_struct
        self.mutations = mutations
        self.ddg = ddg

    def get_descriptor(self, mat, agg=np.mean):
        # MolWeight:FASG760101, Hydrophobic:ARGP820101
        return agg([mat[mut.m] - mat[mut.w] for mut in self.mutations])

    def get_ei(self, mat=BLOSUM62):
        struct = self.struct
        eis = [EI(mut.m, mut.w, struct.get_profile(mut.chain_id), mut.i, mat)
               for mut in self.mutations]
        return np.sum(eis)

    def get_shells_cp(self, inner, outer, mat=BASU010101):
        cps = [CP(mut, self.struct, mat, inner, outer) for mut in self.mutations]
        return np.sum(cps, axis=0)

    def get_asa(self, agg=np.sum):
        func = lambda stride: stride["ASA_Chain"]-stride["ASA"]
        struct = self.struct
        return agg([func(struct.stride[(mut.chain_id, mut.i)])
                    for mut in self.mutations])

    def features(self, free_mem=0):
        if self.struct.dist_mat is None:
            self.struct.compute_dist_mat()
        log_mutations = np.log(len(self.mutations))
        hydphob = self.get_descriptor(ARGP820101)
        molweight = self.get_descriptor(FASG760101)
        tota_asa = self.get_asa()
        ei = self.get_ei()
        cp_a1, cp_b1, _ = self.get_shells_cp(2.0, 4.0)
        cp_a2, cp_b2, _ = self.get_shells_cp(4.0, 6.0)
        if free_mem == 1:
            self.struct.free_dist_mat()
        feats = [log_mutations, hydphob, molweight, tota_asa,
                 ei, cp_a1, cp_b1, cp_a2, cp_b2]
        return np.asarray(feats)


class SkempiStruct(object):

    def __init__(self, modelname, chains_a, chains_b, pdb_path=PDB_PATH):
        fd = open(osp.join(pdb_path, "%s.pdb" % modelname), 'r')
        pdb = modelname[:4].upper()
        struct = parse_pdb(pdb, fd)
        self.struct = PDB(pdb, {c: struct[c] for c in chains_a + chains_b})
        self.modelname = modelname
        self.chains_a = chains_a
        self.chains_b = chains_b
        self.res_chain_to_atom_indices = {}
        self.atom_indices_to_chain_res = {}
        self.atoms = []
        self.init_dictionaries()
        self.dist_mat = None
        self._profiles = {}
        self._alignments = {}
        self.init_profiles()
        self._stride = Stride(pdb)

    @property
    def pdb(self):
        return self.modelname[:4].upper()

    def __str__(self):
        return "<SkempiStruct %s_%s_%s>" % (self.pdb, self.chains_a, self.chains_b)

    def init_profiles(self):
        chains = self.chains
        self._profiles = {c: SkempiProfile(self.pdb, c) for c in chains}
        self._alignments = {c: MSA(self.pdb, c) for c in chains}

    def get_profile(self, chain_id):
        return self._profiles[chain_id]

    def get_alignment(self, chain_id):
        return self._alignments[chain_id]

    @property
    def stride(self):
        return self._stride

    @property
    def chains(self):
        return self.struct.chains

    def init_dictionaries(self):
        for chain in self.struct:
            for res_i, res in enumerate(chain):
                for atom in res:
                    if atom.type != 'C':
                        continue
                    chain_id = chain.chain_id
                    if (chain_id, res_i) in self.res_chain_to_atom_indices:
                        self.res_chain_to_atom_indices[(chain_id, res_i)].append(len(self.atoms))
                    else:
                        self.res_chain_to_atom_indices[(chain_id, res_i)] = [len(self.atoms)]
                    self.atom_indices_to_chain_res[len(self.atoms)] = (chain_id, res_i)
                    self.atoms.append(atom)

    def compute_dist_mat(self):
        self.dist_mat = get_distance_matrix(atoms1=self.atoms, atoms2=self.atoms)

    def free_dist_mat(self):
        del self.dist_mat
        self.dist_mat = None

    def __getitem__(self, chain_id):
        return self.chains[chain_id]

    def _get_indices(self, chain_id, res_i, condition):
        mat = self.dist_mat
        row_indices = self.res_chain_to_atom_indices[(chain_id, res_i)]
        col_indices = []
        for i, row in enumerate(row_indices):
            col_indices.extend([ix for ix in np.where(condition(mat[row]))[0]])
        return set([self.atom_indices_to_chain_res[col_i] for col_i in col_indices])

    def get_sphere_indices(self, chain_id, res_i, threshold):
        return self._get_indices(chain_id, res_i, lambda mat_row: mat_row <= threshold)

    def get_shell_indices(self, chain_id, res_i, r_inner, r_outer):
        return self._get_indices(chain_id, res_i, lambda mat_row: np.logical_and(r_inner <= mat_row, mat_row <= r_outer))

    def __iter__(self):
        for chain_obj in self.chains.values():
            yield chain_obj


def to_fasta(skempi_entries, out_file):
    sequences = []
    for entry in skempi_entries:
        for chain in entry:
            sequences.append(SeqRecord(Seq(chain.seq), chain.id))
    SeqIO.write(sequences, open(out_file, 'w+'), "fasta")


def CP(mut, skempi, C, inner, outer):
    i, chain_a = mut.i, mut.chain_id
    m, w = mut.m, mut.w
    retA, retB = 0, 0
    for chain_id, j in skempi.get_shell_indices(chain_a, i, inner, outer):
        a = skempi[chain_id][j].name
        if j == i and chain_id == chain_a:
            assert a == w
            continue
        if chain_id == chain_a:
            retA += C[(a, m)] - C[(a, w)]
        else:
            retB += C[(a, m)] - C[(a, w)]
    return retA, retB, retA + retB


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


def load_skempi_structs(pdb_path, compute_dist_mat=True):
    prots = skempi_df.Protein.values
    skempi_structs = {}
    for t in tqdm(set([tuple(pdb_str.split('_')) for pdb_str in prots]),
                  desc="skempi structures processed"):
        struct = SkempiStruct(*t, pdb_path=pdb_path)
        if compute_dist_mat:
            struct.compute_dist_mat()
        skempi_structs[t] = struct
    return skempi_structs


def load_skempi_records(skempi_structs, minus_ddg=False):
    records = []
    pbar = tqdm(range(len(skempi_df)), desc="skempi records processed")
    for _, row in skempi_df.iterrows():
        d_row = row.to_dict()
        pdb, ca, cb = tuple(row.Protein.split('_'))
        mutations = [Mutation(s, reverse=minus_ddg)
                     for s in d_row["Mutation(s)_cleaned"].split(',')]
        if minus_ddg:
            mut_strs = ['%d%s%s' % (mut.i + 1, AAA[mut.w], mut.chain_id)
                        for mut in mutations]
            modelname = '%s_%s' % (pdb.lower(), ':'.join(mut_strs))
            try:
                struct = SkempiStruct(modelname, ca, cb, pdb_path='../data/pdbs_r/')
            except IOError as e:
                struct = None
        else:
            struct = skempi_structs[(pdb, ca, cb)]
        ddg = -d_row["DDG"] if minus_ddg else d_row["DDG"]
        record = SkempiRecord(struct, mutations, ddg)
        records.append(record)
        pbar.update(1)
    pbar.close()
    return records


if __name__ == "__main__":

    skempi_entries = list(load_skempi_structs(PDB_PATH).values())
    to_fasta(skempi_entries, "../data/skempi.fas")

    lines = []

    for entry in tqdm(skempi_entries, desc="skempi entries processed"):

        for chain in entry:
            aln = entry.get_alignment(chain.chain_id)
            aln.to_fasta("../data/msas/%s.msa" % chain.id)

            cline = "perl /groups/bioseq.home/Josef/consurf-db/installation/bin/consurf " \
                    "--PDB /groups/bioseq.home/Yotam/skempi/pdbs/%s.pdb " \
                    "--CHAIN %s --Out_Dir /groups/bioseq.home/Yotam/skempi/consurf/%s " \
                    "-MSA /groups/bioseq.home/Yotam/skempi/msas/%s.msa -SEQ_NAME %s " \
                    "-Matrix WAG --w /groups/bioseq.home/Yotam/skempi/consurf/%s --consurfdb\n" \
                    % (chain.pdb, chain.chain_id, chain.id, chain.id, chain.id, chain.id)

            lines.append(cline)

    with open("../data/consurf.sh", "w+") as f:
        f.writelines(lines)
