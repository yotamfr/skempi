import os
import sys

import numpy as np
import pandas as pd
import os.path as osp

import stride
from modeller import *
from aaindex import *
from pdb_utils import *
from grid_utils import *
from skempi_consts import *


class MSA(object):
    def __init__(self, pdb, chain):
        uid = "%s_%s.aln" % (pdb, chain)
        with open(osp.join("..", "data", "skempi_aln", uid), 'r') as f:
            lines = f.readlines()
        kvs = [[s for s in line.split(' ') if s] for line in lines]
        msa = [(k.strip(), v.strip()) for k, v in kvs]
        self.ali = np.asarray([list(seq) for _, seq in msa])

    def M(self, r1, r2):
        # return BLOSUM62[(r1, r2)]
        return iPTMs1[(r1, r2)]

    def N(self, aa, i):
        return sum(self.ali[:, i] == aa)

    def Np(self, aa, i, params):
        a, b, c = params
        if b == 0 and c == 0:
            return a
        elif c == 0:
            return a + b * self.n_gap(i)
        elif b == 0:
            a + c * self.E(aa, i)
        else:
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


class Profile(object):
    def __init__(self, pdb, chain):
        uid = "%s_%s.prof" % (pdb, chain)
        df = pd.read_csv(osp.join("..", "data", "skempiprofiles", uid), sep=' ')
        self._profile = [df.loc[i].to_dict() for i in range(len(df))]

    def __getitem__(self, t):
        i, a = t
        pos_dict = self._profile[i]
        return pos_dict[a]


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


def get_stride(pdb_struct, ca, cb):
    modelname = pdb_struct.pdb
    pdb_pth = osp.join('stride', modelname, '%s.pdb' % modelname)
    out_pth = osp.join('stride', modelname, '%s.out' % modelname)
    if not osp.exists(osp.dirname(pdb_pth)):
        os.makedirs(osp.dirname(pdb_pth))
    if not osp.exists(out_pth):
        pdb_struct.to_pdb(pdb_pth)
        stride.main(pdb_pth, ca, cb, out_pth)
    return Stride(pd.read_csv(out_pth))


class SkempiStruct(PDB):

    def __init__(self, pdb_struct, chains_a, chains_b):
        cs = list(chains_a + chains_b)
        super(SkempiStruct, self).__init__(pdb_struct.pdb,
                                           pdb_struct.atoms,
                                           pdb_struct._chains,
                                           dict(zip(cs, cs)))
        self._stride = get_stride(self, chains_a, chains_b)

    @property
    def stride(self):
        return self._stride

    @property
    def modelname(self):
        return self.pdb


class SkempiRecord(object):

    def __init__(self, struct, chains_a, chains_b, mutations,
                 load_mutant=True, init_profiles=True):
        self.struct = SkempiStruct(struct, chains_a, chains_b)
        self.mutant = None
        self.chains_a = chains_a
        self.chains_b = chains_b
        self.mutations = mutations
        self._profiles = None
        self._alignments = None
        if load_mutant:
            self.init_mutant()
        if init_profiles:
            self.init_profiles()
            self.init_alignments()

    @property
    def pdb(self):
        return self.struct.pdb[:4].upper()

    def init_profiles(self):
        if self._profiles is not None:
            return
        self._profiles = {c: Profile(self.pdb, c) for c in self.chains}

    def init_alignments(self):
        if self._alignments is not None:
            return
        self._alignments = {c: MSA(self.pdb, c) for c in self.chains}

    def get_profile(self, chain_id):
        return self._profiles[chain_id]

    def get_alignment(self, chain_id):
        return self._alignments[chain_id]

    @property
    def chains(self):
        return self.struct.chains

    def features(self):
        return get_features(self)

    def init_mutant(self):
        wt = self.struct
        mutantname, ws = apply_modeller(wt, self.mutations)
        pth = osp.join(ws, "%s.pdb" % mutantname)
        mutant_sturct = parse_pdb(mutantname, open(pth, 'r'))
        self.mutant = SkempiStruct(mutant_sturct, self.chains_a, self.chains_b)

    def __reversed__(self):
        mutations = [reversed(mut) for mut in self.mutations]
        record = SkempiRecord(self.mutant, self.chains_a, self.chains_b, mutations,
                              load_mutant=False, init_profiles=False)
        record._profiles = self._profiles
        record._alignments = self._alignments
        record.mutant = self.struct
        return record

    def __str__(self):
        return "<%s_%s_%s: %s>" % (self.pdb, self.chains_a, self.chains_a, [str(mut) for mut in self.mutations])

    def __hash__(self):
        return hash(str(self))


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


def get_descriptor(mutations, mat, agg=np.mean):
    return agg([mat[mut.m] - mat[mut.w] for mut in mutations])


def EI(m, w, P, i, B):
    return sum([P[(i, a)] * (B[(a, m)] - B[(a, w)]) for a in amino_acids])


def get_EIs(record, mat=BLOSUM62):
    return [EI(mut.m, mut.w, record.get_profile(mut.chain_id), mut.i, mat) for mut in record.mutations]


def comp_CP_in_shell(mut, struct, mat, inner, outer):
    res = struct[mut.chain_id][mut.i]
    cp_A, cp_B = get_cp_in_shell_around_res(res, struct.atoms, inner, outer, mat, ['C'])
    return cp_A[mut.m], cp_B[mut.m]


def get_CPs_shell(record, inner, outer, mat=BASU010101):
    return [comp_CP_in_shell(mut, record.struct, mat, inner, outer) for mut in record.mutations]


def comp_CP_in_sphere(mut, struct, mat, rad):
    res = struct[mut.chain_id][mut.i]
    cp_A, cp_B = get_cp_in_sphere_around_res(res, struct.atoms, rad, mat, ['C'])
    return cp_A[mut.m], cp_B[mut.m]


def get_CPs_sphere(record, rad, mat=BASU010101):
    return [comp_CP_in_sphere(mut, record.struct, mat, rad) for mut in record.mutations]


def EVO(record, mut, a=25, b=15, c=5):
    i, w, m = mut.i, mut.w, mut.m
    ali = record.get_alignment(mut.chain_id)
    nom = ali.N(m, i) + ali.Np(m, i, (a, b, c))
    denom = ali.N(w, i) + ali.Np(w, i, (a, b, c))
    return -np.log(float(nom) / denom)


def get_EVOs(record, a, b, c):
    return [EVO(record, mut,  a, b, c) for mut in record.mutations]


def HSE(struct, mut, rad):
    res = struct[mut.chain_id][mut.i]
    hse_up, hse_down = get_hse_in_sphere(res, struct.atoms, rad, ['C'])
    return hse_up, hse_down


def get_HSEs(record, rad):
    hses_mut = [HSE(record.mutant, mut, rad) for mut in record.mutations]
    hses_wt = [HSE(record.struct, mut, rad) for mut in record.mutations]
    return [np.log((1.0 + len(up_mut) * len(down_wt)) / (1.0 + len(up_wt) * len(down_mut)))
            for (up_mut, down_mut), (up_wt, down_wt) in zip(hses_mut, hses_wt)]


def get_deltas2(record):
    def delta_asa(mut): return MaxASA_emp[mut.w] - MaxASA_emp[mut.m]
    return [delta_asa(mut) for mut in record.mutations]


def get_deltas1(record):
    def delta_asa(stride): return stride["ASA_Chain"] - stride["ASA"]
    deltas = [delta_asa(record.mutant.stride[(m.chain_id, m.i)])-
              delta_asa(record.struct.stride[(m.chain_id, m.i)])
              for m in record.mutations]
    return deltas


def get_counts(record):
    ret = {AAA_dict[aa]: 0.0 for aa in amino_acids}
    for mut in record.mutations:
        ret[AAA_dict[mut.w]] += 1
        ret[AAA_dict[mut.m]] -= 1
    return ret


def avg_bfactor(struct, mut):
    res_i, chain_id = mut.i, mut.chain_id
    res = struct[chain_id][res_i]
    temps = [a.temp for a in res.atoms]
    return np.mean(temps)


def get_bfactor(record, agg=np.min, pdb_path="../data/pdbs_n"):  # obtain wt b-factor
    struct = parse_pdb(record.pdb, open(osp.join(pdb_path, "%s.pdb" % record.pdb), 'r'))
    return agg([avg_bfactor(struct, mut) for mut in record.mutations])


def parse_mutations(mutations_str, reverse=False, sep=','):
    mutations = [Mutation(s) for s in mutations_str.split(sep)]
    if reverse:
        return [reversed(mut) for mut in mutations]
    else:
        return mutations


def get_features(record):
    feats = dict()
    feats["hydphob"] = get_descriptor(record.mutations, ARGP820101)
    feats["molweight"] = get_descriptor(record.mutations, FASG760101)
    feats["evo"] = sum(get_EVOs(record, a=50, b=0, c=0))
    feats["delta_asa1"] = sum(get_deltas1(record))
    feats["delta_asa2"] = np.mean(get_deltas2(record))
    feats["hse"] = sum(get_HSEs(record, 6.0))
    feats["ei"] = sum(get_EIs(record))
    feats["bfactor"] = get_bfactor(record)
    feats.update(get_counts(record))
    return feats


def load_skempi_v1(load_mutants=True):
    from tqdm import tqdm
    structs = {}
    for i in tqdm(range(len(skempi_df)), "loading structures"):
        row = skempi_df.loc[i]
        t = tuple(row.Protein.split('_'))
        pth = osp.join("..", "data", "pdbs", "%s.pdb" % t[0])
        structs[t] = parse_pdb(t[0], open(pth, 'r'))
    records = []
    for i in tqdm(range(len(skempi_df)), "loading records"):
        row = skempi_df.loc[i]
        mutations = parse_mutations(row["Mutation(s)_cleaned"])
        modelname, chain_A, chain_B = t = row.Protein.split('_')
        r = SkempiRecord(structs[tuple(t)], chain_A, chain_B, mutations, load_mutants)
        records.append(r)
    return records


if __name__ == "__main__":
    from tqdm import tqdm
    structs = {}
    i = 100
    row = skempi_df.loc[i]
    t = tuple(row.Protein.split('_'))
    pth = osp.join("..", "data", "pdbs", "%s.pdb" % t[0])
    structs[t] = parse_pdb(t[0], open(pth, 'r'))
    records = []
    row = skempi_df.loc[i]
    mutations = parse_mutations(row["Mutation(s)_cleaned"])
    modelname, chain_A, chain_B = t = row.Protein.split('_')
    r = SkempiRecord(structs[tuple(t)], chain_A, chain_B, mutations)
    f = r.features()
    print(f)
