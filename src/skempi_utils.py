import os
import sys
import pickle

import numpy as np
import pandas as pd
import os.path as osp
from tqdm import tqdm

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import manhattan_distances

try:
    import src.stride
    from src.consts import *
    from src.aaindex import *
    from src.modeller import *
    from src.pdb_utils import *
    from src.bindprofx import *
    from src.skempi_consts import *
except ImportError:
    import stride
    from consts import *
    from aaindex import *
    from modeller import *
    from pdb_utils import *
    from bindprofx import *
    from skempi_consts import *


def get_distance_matrix(atoms1, atoms2, metric):
    X = np.matrix([a.coord for a in atoms1])
    Y = np.matrix([a.coord for a in atoms2])
    return metric(X, Y)


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


class MSA(object):
    def __init__(self, pdb, chain):
        uid = "%s_%s.aln" % (pdb, chain)
        with open(osp.join("..", "data", "skempi_aln", uid), 'r') as f:
            lines = f.readlines()
        kvs = [[s for s in line.split(' ') if s] for line in lines]
        msa = [(k.strip(), v.strip()) for k, v in kvs]
        self.ali = np.asarray([list(seq) for _, seq in msa])
        df = pd.read_csv("../data/iPTMs1.txt", sep=" ")
        self.iPTMs1 = {}
        for r1, r2 in comb(amino_acids, 2):
            self.iPTMs1[(r1, r2)] = df[df.AA == r1][r2].values[0]
            self.iPTMs1[(r2, r1)] = df[df.AA == r2][r1].values[0]
            self.iPTMs1[(r1, r1)] = df[df.AA == r1][r1].values[0]
            self.iPTMs1[(r2, r2)] = df[df.AA == r2][r2].values[0]

    def M(self, r1, r2):
        return self.iPTMs1[(r1, r2)]

    def N(self, aa, i):
        return sum(self.ali[:, i] == aa)

    def Np(self, aa, i, params):
        a, b, c = params
        return a + b * self.n_gap(i) + c * self.E(aa, i)

    def E(self, aa, i):
        return sum([self.P(aa, i) * self.M(r, aa) for r in amino_acids])

    def P(self, aa, i):
        return self.N(aa, i) / float(len(self))

    def n_gap(self, i):
        return self.N('-', i)

    def __len__(self):
        return len(self.ali)

    def __getitem__(self, i):
        return self.ali[i]

    def to_fasta(self, pth):
        lines = [">%s\n%s\n" % (uid if i == 0 else "SEQ%d" % i, seq)
                 for i, (uid, seq) in enumerate(self._msa)]
        with open(pth, "w+") as f:
            f.writelines(lines)


class Profile(object):
    def __init__(self, pdb, chain):
        uid = "%s_%s.prof" % (pdb, chain)
        df = pd.read_csv(osp.join("..", "data", "skempiprofiles", uid), sep=' ')
        self._profile = [df.loc[i].to_dict() for i in range(len(df))]

    def __getitem__(self, t):
        i, a = t
        pos_dict = self._profile[i]
        return pos_dict[a]  ### + 0.05 * pos_dict['-']


# class Profile(object):
#     def __init__(self, pdb, chain):
#         uid = "%s_%s" % (pdb, chain)
#         doc = collection_msa.find_one({
#             "_id": uid})
#         assert doc is not None
#         self._profile = doc["profile"]
#
#     def __getitem__(self, t):
#         i, a = t
#         return self._profile[i][a]


class Stride(object):
    def __init__(self, stride_df):
        self._dict = {}
        self._total = 0.0
        for i, row in stride_df.iterrows():
            d_row = row.to_dict()
            try:
                chain_id = d_row["Chain"]
                res_i = int(d_row["Res"]) - 1
            except ValueError as e:
                continue
            self._total += (d_row["ASA_Chain"] - d_row["ASA"])
            self._dict[(chain_id, res_i)] = d_row

    def __getitem__(self, t):
        chain_id, res_i = t
        return self._dict[(chain_id, res_i)]


class SkempiRecord(object):
    def __init__(self, skempi_struct, mutations, ddg, group=0, minus_ddg=False):
        self.struct = skempi_struct
        self.mutations = mutations
        self.is_minus = minus_ddg
        self.group = group
        self.ddg = ddg

    @property
    def modelname(self):
        return self.struct.modelname

    def __hash__(self):
        return hash((self.struct, tuple(self.mutations)))

    def get_ei(self, mat=BLOSUM62):
        struct = self.struct
        eis = [EI(mut.m, mut.w, struct.get_profile(mut.chain_id), mut.i, mat) for mut in self.mutations]
        return np.sum(eis)

    def get_evo(self):
        struct = self.struct
        scores = [EVO(struct.get_alignment(mut.chain_id), mut) for mut in self.mutations]
        return np.sum(scores)

    def get_shells_cp(self, inner, outer, mat=BASU010101):
        cps = [CP(mut, self.struct, mat, inner, outer) for mut in self.mutations]
        return np.sum(cps, axis=0)

    def get_asa(self, agg=np.sum):
        return agg([delta_asa(self.struct, mut) for mut in self.mutations])

    def features(self, free_mem=True):
        try:
            self.struct.init_profiles()
            self.struct.init_alignments()
            self.struct.init_stride()
        except IOError as e:
            print("warining: %s" % e)
        if self.struct.dist_mat is None:
            self.struct.compute_dist_mat()
        log_mutations = np.log(len(self.mutations))
        bfactor = self.get_bfactor()
        hydphob = get_descriptor(self.mutations, ARGP820101)
        molweight = get_descriptor(self.mutations, FASG760101)
        tota_asa = self.get_asa()
        ei = self.get_ei()
        evo = self.get_evo()
        cp_a1, cp_b1, _ = self.get_shells_cp(2.0, 4.0)
        cp_a2, cp_b2, _ = self.get_shells_cp(4.0, 6.0)
        if free_mem: self.struct.free_dist_mat()
        feats = [log_mutations, bfactor, hydphob, molweight, tota_asa, ei, evo, cp_a1, cp_b1, cp_a2, cp_b2]
        return np.asarray(feats)

    def get_bfactor(self, agg=np.min, pdb_path="../data/pdbs_n"):  # obtain wt b-factor
        pdb = self.struct.pdb
        ca = self.struct.chains_a
        cb = self.struct.chains_b
        struct = SkempiStruct(pdb, ca, cb, pdb_path=pdb_path)
        return agg([avg_bfactor(struct, mut) for mut in self.mutations])

    def __reversed__(self):
        wt = self.struct
        co = self.struct.carbons_only
        mutations = [reversed(mut) for mut in self.mutations]
        modelname, ws = apply_modeller(wt.struct, self.mutations)
        mutant = SkempiStruct(modelname, wt.chains_a, wt.chains_b, pdb_path=ws, carbons_only=co)
        return SkempiRecord(mutant, mutations, -self.ddg, self.group, not self.is_minus)

    def __str__(self):
        return "<%s: %s>" % (self.struct, [str(m) for m in self.mutations])


class SkempiStruct(object):

    def __init__(self, modelname, chains_a, chains_b, pdb_path=PDB_PATH, carbons_only=True):
        self.path = osp.join(pdb_path, "%s.pdb" % modelname)
        cs = list(chains_a + chains_b)
        self.struct = parse_pdb(modelname, open(self.path, 'r'), dict(zip(cs, cs)))
        self.modelname = modelname
        self.carbons_only = carbons_only
        self.chains_a = chains_a
        self.chains_b = chains_b
        self.res_chain_to_atom_indices = {}
        self.atom_indices_to_chain_res = {}
        self.atoms = []
        self.init_dictionaries()
        self.dist_mat = None
        self._profiles = None
        self._alignments = None
        self._stride = None

    def __hash__(self):
        return hash((self.modelname, self.chains_a, self.chains_b))

    @property
    def pdb(self):
        return self.modelname[:4].upper()

    def __str__(self):
        return "<SkempiStruct %s_%s_%s>" % (self.modelname, self.chains_a, self.chains_b)

    def init_profiles(self):
        if self._profiles is not None:
            return
        self._profiles = {c: Profile(self.pdb, c) for c in self.chains}

    def init_alignments(self):
        if self._alignments is not None:
            return
        self._alignments = {c: MSA(self.pdb, c) for c in self.chains}

    def init_stride(self):
        ca = self.chains_a
        cb = self.chains_b
        pdb_struct = self.struct
        modelname = pdb_struct.pdb
        pdb_pth = osp.join('stride', modelname, '%s.pdb' % modelname)
        out_pth = osp.join('stride', modelname, '%s.out' % modelname)
        if not osp.exists(osp.dirname(pdb_pth)):
            os.makedirs(osp.dirname(pdb_pth))
        if not osp.exists(out_pth):
            pdb_struct.to_pdb(pdb_pth)
            stride.main(pdb_pth, ca, cb, out_pth)
        self._stride = Stride(pd.read_csv(out_pth))

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
        carbons_only = self.carbons_only
        for chain in self.struct:
            for res_i, res in enumerate(chain):
                for atom in res:
                    if carbons_only and atom.type != 'C':
                        continue
                    chain_id = chain.chain_id
                    if (chain_id, res_i) in self.res_chain_to_atom_indices:
                        self.res_chain_to_atom_indices[(chain_id, res_i)].append(len(self.atoms))
                    else:
                        self.res_chain_to_atom_indices[(chain_id, res_i)] = [len(self.atoms)]
                    self.atom_indices_to_chain_res[len(self.atoms)] = (chain_id, res_i)
                    self.atoms.append(atom)

    def compute_dist_mat(self, metric=euclidean_distances):
        X = np.matrix([a.coord for a in self.atoms])
        self.dist_mat = metric(X, X)

    def free_dist_mat(self):
        del self.dist_mat
        self.dist_mat = None

    def __getitem__(self, chain_id):
        return self.chains[chain_id]

    def _get_indices(self, chain_id, res_i, condition):
        mat = self.dist_mat
        if mat is None:
            raise ValueError("call: struct.compute_dist_mat()")
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
        for chain_obj in self.struct:
            yield chain_obj


def to_fasta(skempi_entries, out_file):
    lines = []
    for entry in skempi_entries:
        lines.extend([">%s\n%s\n" % (chain.id, chain.seq) for i, chain in enumerate(entry)])
    with open(out_file, "w+") as f:
        f.writelines(lines)


def delta_asa(struct, mut):
    stride = struct.stride[(mut.chain_id, mut.i)]
    return stride["ASA_Chain"] - stride["ASA"]


def avg_bfactor(struct, mut):
    res_i, chain_id = mut.i, mut.chain_id
    res = struct[chain_id][res_i]
    temps = [a.temp for a in res.atoms]
    return np.mean(temps)


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


def get_descriptor(mutations, mat, agg=np.mean):    # MolWeight:FASG760101, Hydrophobic:ARGP820101
    return agg([mat[mut.m] - mat[mut.w] for mut in mutations])


def EI(m, w, P, i, B):
    return sum([P[(i, a)] * (B[(a, m)] - B[(a, w)]) for a in amino_acids])


def EVO(ali, mut, a=25, b=15, c=5):
    i, w, m = mut.i, mut.w, mut.m
    nom = ali.N(m, i) + ali.Np(m, i, (a, b, c))
    denom = ali.N(w, i) + ali.Np(w, i, (a, b, c))
    return -np.log(nom / denom)


def save_object(obj, filename):
    with open(filename, 'w+b') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def load_object(pth):
    with open(pth, 'rb') as f:
        loaded_dist_mat = pickle.load(f)
        assert len(loaded_dist_mat) > 0
    return loaded_dist_mat


def load_skempi_structs(pdb_path, compute_dist_mat=False, carbons_only=True):
    prots = skempi_df.Protein.values
    skempi_structs = {}
    for t in tqdm(set([tuple(pdb_str.split('_')) for pdb_str in prots]),
                  desc="skempi structures processed"):
        struct = SkempiStruct(*t, pdb_path=pdb_path, carbons_only=carbons_only)
        if compute_dist_mat: struct.compute_dist_mat()
        skempi_structs[t] = struct
    return skempi_structs


def parse_mutations(mutations_str, reverse=False, sep=','):
    mutations = [Mutation(s) for s in mutations_str.split(sep)]
    if reverse:
        return [reversed(mut) for mut in mutations]
    else:
        return mutations


def load_skempi_records(skempi_structs):
    records = []
    pbar = tqdm(range(len(skempi_df)), desc="skempi records processed")
    for _, row in skempi_df.iterrows():
        d_row = row.to_dict()
        pdb, ca, cb = tuple(row.Protein.split('_'))
        mutations = parse_mutations(d_row["Mutation(s)_cleaned"])
        struct = skempi_structs[(pdb, ca, cb)]
        ddg = d_row["DDG"]
        groups = ["%s_%s_%s" % (pdb, ca, cb) in g for g in [G1, G2, G3, G4, G5]]
        assert sum(groups) <= 1
        group = 0 if sum(groups) == 0 else (groups.index(True) + 1)
        record = SkempiRecord(struct, mutations, ddg, group)
        records.append(record)
        pbar.update(1)
    pbar.close()
    return records


def records_to_xy(skempi_records, load_negative=False):
    data = []
    flag = int(load_negative)
    for record in tqdm(skempi_records, desc="records processed"):
        assert record.struct is not None
        r = reversed(record) if load_negative else record
        data.append([r.features(True), r.ddg, [r.group, flag], r.modelname, r.mutations])
    return data


def skempi_group_from_row(row):
    pdb, ca, cb = row.Protein.split('_')
    groups = ["%s_%s_%s" % (pdb, ca, cb) in g for g in [G1, G2, G3, G4, G5]]
    assert sum(groups) <= 1
    return 0 if sum(groups) == 0 else (groups.index(True) + 1)


def skempi_record_from_row(row, pdb_path='../data/pdbs', carbons_only=False):
    t = row.Protein.split('_')
    struct = SkempiStruct(*t, pdb_path=pdb_path, carbons_only=carbons_only)
    mutations = parse_mutations(row["Mutation(s)_cleaned"])
    ddg = row.DDG
    group = skempi_group_from_row(row)
    return SkempiRecord(struct, mutations, ddg, group)


def consurf():
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


if __name__ == "__main__":
    for i in tqdm(range(len(skempi_df))):
        row = skempi_df.loc[i]
        r = skempi_record_from_row(row, pdb_path="../data/pdbs_n")
        r.features(True)
        print(bindprofx(r))
        rr = reversed(r)
        rr.features(True)
        # print(bindprofx(rr))
