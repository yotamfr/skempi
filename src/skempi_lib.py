import os
import sys

from tqdm import tqdm
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
        # df = pd.read_csv(osp.join("..", "data", 'skempiprofiles', uid), sep=' ')
        df = pd.read_csv(osp.join("..", "hhblits", uid), sep=',')
        self._profile = [df.loc[i].to_dict() for i in range(len(df))]

    def __getitem__(self, t):
        i, a = t
        pos_dict = self._profile[i]
        return pos_dict[a]

    def __len__(self):
        return len(self._profile)


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

    def __init__(self, pdb_struct, chains_a, chains_b, profiles=None, alignments=None):
        cs = list(chains_a + chains_b)
        super(SkempiStruct, self).__init__(pdb_struct.pdb,
                                           pdb_struct.atoms,
                                           pdb_struct._chains,
                                           dict(zip(cs, cs)))
        self.profiles = profiles
        self.alignments = alignments
        self.init_profiles()
        self.init_alignments()
        self.stride = get_stride(self, chains_a, chains_b)

    def init_profiles(self):
        if self.profiles is not None:
            return
        self.profiles = {c: Profile(self.pdb, c) for c in self.chains}

    def init_alignments(self):
        if self.alignments is not None:
            return
        self.alignments = {c: MSA(self.pdb, c) for c in self.chains}

    def get_profile(self, chain_id):
        return self.profiles[chain_id]

    def get_alignment(self, chain_id):
        return self.alignments[chain_id]

    @property
    def modelname(self):
        return self.pdb


class SkempiRecord(object):

    def __init__(self, skempi_struct, chains_a, chains_b, mutations, ddg=None, load_mutant=True):
        self.struct = skempi_struct
        self.mutant = None
        self.chains_a = chains_a
        self.chains_b = chains_b
        self.mutations = mutations
        self.ddg = ddg
        if load_mutant:
            self.init_mutant()

    @property
    def pdb(self):
        return self.struct.pdb[:4].upper()

    def init_mutant(self):
        wt = self.struct
        mutantname, ws = apply_modeller(wt, self.mutations)
        pth = osp.join(ws, "%s.pdb" % mutantname)
        mutant_sturct = parse_pdb(mutantname, open(pth, 'r'))
        profiles, alignments = self.struct.profiles, self.struct.alignments
        self.mutant = SkempiStruct(mutant_sturct, self.chains_a, self.chains_b, profiles, alignments)

    def __reversed__(self):
        mutations = [reversed(mut) for mut in self.mutations]
        record = SkempiRecord(self.mutant, self.chains_a, self.chains_b, mutations, -self.ddg, load_mutant=False)
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


def parse_mutations(mutations_str, reverse=False, sep=','):
    mutations = [Mutation(s) for s in mutations_str.split(sep)]
    if reverse:
        return [reversed(mut) for mut in mutations]
    else:
        return mutations


def comp_cp_in_shell(mut, struct, mat, inner, outer):
    res = struct[mut.chain_id][mut.i]
    cp_A, cp_B = get_cp_in_shell_around_res(res, struct.atoms, inner, outer, mat, ['C'])
    return cp_A[mut.m], cp_B[mut.m]


def get_cps_shell(struct, mutations, inner, outer, mat=BASU010101):
    return [comp_cp_in_shell(mut, struct, mat, inner, outer) for mut in mutations]


def comp_cp_in_sphere(mut, struct, mat, rad):
    res = struct[mut.chain_id][mut.i]
    cp_A, cp_B = get_cp_in_sphere_around_res(res, struct.atoms, rad, mat, ['C'])
    return cp_A[mut.m], cp_B[mut.m]


def get_cps_sphere(record, rad, mat=BASU010101):
    return [comp_cp_in_sphere(mut, record.struct, mat, rad) for mut in record.mutations]


def avg_bfactor(struct, mut):
    res_i, chain_id = mut.i, mut.chain_id
    res = struct[chain_id][res_i]
    temps = [a.temp for a in res.atoms]
    return np.mean(temps)


def get_bfactor(record, agg=np.min, pdb_path="../data/pdbs_n"):  # obtain wt b-factor
    struct = parse_pdb(record.pdb, open(osp.join(pdb_path, "%s.pdb" % record.pdb), 'r'))
    return agg([avg_bfactor(struct, mut) for mut in record.mutations])


def get_counts(mutations):
    ret = {AAA_dict[aa]: 0.0 for aa in amino_acids}
    for mut in mutations:
        ret[AAA_dict[mut.w]] += 1.0
        ret[AAA_dict[mut.m]] -= 1.0
    return ret


def score_cp(struct, mut, inner, outer, mat=BASU010101):
    w, m = mut.w, mut.m
    center_res = struct[mut.chain_id][mut.i]
    center = center_res.ca.coord
    neighbors = get_atoms_in_shell_around_center(center, struct.atoms, inner, outer, ['C'])
    residues_indices = set([(a.res.chain_id, a.res.index) for a in neighbors
                            if a.res.chain.chain_id != mut.chain_id])
    return sum([sum([struct.get_profile(A)[(j, a)] * (mat[(a, m)] - mat[(a, w)])
                for a in amino_acids]) for A, j in residues_indices])


def score_cp46(struct, mut, mat=BASU010101):
    return score_cp(struct, mut, 4.0, 6.0, mat=mat)


def score_cp68(struct, mut, mat=BASU010101):
    return score_cp(struct, mut, 6.0, 8.0, mat=mat)


def score_bv(struct, mut, PBV=BASU010101, rad=8.0):
    w, m = mut.w, mut.m
    center_res = struct[mut.chain_id][mut.i]
    center = center_res.ca.coord
    neighbors = get_atoms_in_sphere_around_center(center, struct.atoms, rad, ['C'])
    residues_indices = set([(a.res.chain_id, a.res.index) for a in neighbors])
    return sum([sum([struct.get_profile(A)[(j, a)] * (PBV[(a, m)] - PBV[(a, w)])
                     for a in amino_acids]) for A, j in residues_indices])


def score_sk(struct, mut, PSK=SKOJ970101, k=2):
    prof = struct.get_profile(mut.chain_id)
    return sum([sum([prof[(j, a)] * (PSK[(a, mut.m)] - PSK[(a, mut.w)]) for a in amino_acids
                     ]) for j in range(max(0, mut.i-k), min(len(prof), mut.i+k+1)) if j != 0])


def score_evo(struct, mut, a=25, b=0, c=0):
    i, w, m = mut.i, mut.w, mut.m
    ali = struct.get_alignment(mut.chain_id)
    nom = ali.N(m, i) + ali.Np(m, i, (a, b, c))
    denom = ali.N(w, i) + ali.Np(w, i, (a, b, c))
    return -np.log(float(nom) / denom)


def ac_ratio(struct, mut):
    def dasa(stride): return stride["ASA_Chain"] - stride["ASA"]
    d = dasa(struct.stride[(mut.chain_id, mut.i)])
    res = struct[mut.chain_id][mut.i]
    return float(d) / MaxASA_emp[res.name]


def score_bi(struct, mut, B=BLOSUM62):
    prof = struct.get_profile(mut.chain_id)
    return sum([prof[(mut.i, a)] * (B[(a, mut.m)] - B[(a, mut.w)]) for a in amino_acids])


def get_descriptor(mut, mat):
    return mat[mut.m] - mat[mut.w]


def score_descriptor(struct, mut, K):
    prof = struct.get_profile(mut.chain_id)
    return prof[(mut.i, mut.m)] * K[mut.m] - prof[(mut.i, mut.w)] * K[mut.w]


def score_hp(struct, mut):
    return score_descriptor(struct, mut, K=KYTJ820101)


def score_molweight(struct, mut):
    return score_descriptor(struct, mut, K=FASG760101)


def score_hydphob(struct, mut):
    return score_descriptor(struct, mut, K=ARGP820101)


def score_asa(struct, mut):
    return score_descriptor(struct, mut, K=MaxASA_emp)


def agg_multiple_scores_to_one(scores):
    return np.max(scores) + np.min(scores) - np.mean(scores)


def compute_score(score_func, struct, mutations, agg=agg_multiple_scores_to_one, with_ac=True):
    if with_ac:
        return agg([ac_ratio(struct, mut) * score_func(struct, mut) for mut in mutations])
    else:
        return agg([score_func(struct, mut) for mut in mutations])


def get_features(struct, mutations):
    feats = dict()
    feats["Hp"] = compute_score(score_hp, struct, mutations, agg=np.mean)
    feats["MolWeight"] = compute_score(score_molweight, struct, mutations, agg=np.mean)
    feats["MaxASA"] = compute_score(score_asa, struct, mutations, agg=np.mean)
    feats["EVO"] = compute_score(score_evo, struct, mutations, agg=np.sum)
    feats["BI"] = compute_score(score_bi, struct, mutations, agg=np.sum)
    feats.update(get_counts(mutations))
    return feats


def load_skempi(skempi_df, path_to_pdbs, load_mut=True):
    structs = {}
    for i in tqdm(range(len(skempi_df)), "loading structures"):
        row = skempi_df.loc[i]
        modelname, chains_a, chains_b = t = tuple(row.Protein.split('_'))
        if t in structs:
            continue
        pth = osp.join(path_to_pdbs, "%s.pdb" % t[0])
        st = parse_pdb(modelname, open(pth, 'r'))
        structs[t] = SkempiStruct(st, chains_a, chains_b)
    records = []
    for i in tqdm(range(len(skempi_df)), "loading records"):
        row = skempi_df.loc[i]
        mutations = parse_mutations(row["Mutation(s)_cleaned"])
        ddg = row["DDG"]
        modelname, chains_a, chains_b = t = tuple(row.Protein.split('_'))
        r = SkempiRecord(structs[t], chains_a, chains_b, mutations, ddg, load_mutant=load_mut)
        records.append(r)
    return records


def load_skempi_v1():
    return load_skempi(skempi_df, PDB_PATH, True)


def load_skempi_v2():
    return load_skempi(prepare_skempi_v2(), SKMEPI2_PDBs, True)


def prepare_zemu():
    df = pd.read_csv(osp.join('..', 'data', 'dataset_ZEMu.2.csv'))
    df2 = prepare_skempi_v2()
    df1 = skempi_df
    df["DDG"] = df[" Gexp (kcal/mol)"]
    df["ZEMu"] = df[" G ZEMu (kcal/mol) \tID"]
    df["Mutation(s)_cleaned"] = df[" Mutant"].apply(lambda s: s.replace(".", ","))
    pdbs, _, _ = zip(*[prot.split('_') for prot in df2["#Pdb"]])
    D = {pdb: cpx for pdb, cpx in zip(pdbs, df2["#Pdb"])}
    pdbs, _, _ = zip(*[prot.split('_') for prot in df1.Protein])
    D.update({pdb: cpx for pdb, cpx in zip(pdbs, df1.Protein)})
    df["Protein"] = [D[p] for p in df["\PDB ID"]]
    return df


def prepare_skempi_v2():
    with open(osp.join('..', 'data', 'skempi_v2.csv'), 'r') as f:
        lines = f.readlines()
    head = lines.pop(0)
    data, num_mutations = [], []
    while lines:
        line = lines.pop(0)
        row = [s for s in line.strip().split(";")]
        if "1KBH" == row[0][:4]:
            continue
        for i, item in enumerate(row[1:]):
            try: parse_mutations(item)
            except: break
        mutations1 = ",".join(row[1:1 + int(i / 2)])
        mutations2 = ",".join(row[1 + int(i / 2):1 + i])
        assert len(mutations1.split(",")) == len(mutations2.split(","))
        num_mutations.append(len(mutations1.split(",")))
        vals = row[:1] + [mutations1, mutations2] + row[i + 1:]
        data.append(vals)
    cols = head.split(';')
    df = pd.DataFrame(data, columns=cols)
    df["num_mutations"] = num_mutations
    df["Protein"] = df["#Pdb"]
    df = df[np.logical_and(df["Affinity_mut_parsed"] != '', df["Affinity_wt_parsed"] != '')]
    T = df["Temperature"].replace(r'\(assumed\)', '', regex=True).replace(r'', '298', regex=False).astype(float).values
    R = (8.314 / 4184)
    df["dGwt"] = R * T * np.log(df["Affinity_wt_parsed"].astype(float).values)
    df["dGmut"] = R * T * np.log(df["Affinity_mut_parsed"].astype(float).values)
    df["DDG"] = df["dGmut"] - df["dGwt"]
    df.to_csv(osp.join('..', 'data', 'skempi_v2.1.csv'), index=False)
    return df.reset_index(drop=True)


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
    st = structs[tuple(t)]
    r = SkempiRecord(st, chain_A, chain_B, mutations)
    f = get_features(r.struct, r.mutations)
    print(f)
