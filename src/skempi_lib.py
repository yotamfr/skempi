import os
import sys

from tqdm import tqdm
import numpy as np
import pandas as pd
import os.path as osp

from stride import *
from modeller import *
from aaindex import *
from pdb_utils import *
from grid_utils import *
from skempi_consts import *
from hhblits import *
from bindprofx import foldx4


class MSA(object):
    def __init__(self, pdb, chain):
        uid = "%s_%s.aln" % (pdb, chain)
        with open(osp.join("hhblits", uid), 'r') as f:
            lines = f.readlines()
        kvs = [[s for s in line.split(' ') if s] for line in lines]
        msa = [(k.strip(), v.strip()) for k, v in kvs]
        self.ali = np.asarray([list(seq) for _, seq in msa])

    def M(self, r1, r2):
        return BLOSUM62[(r1, r2)]

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
        return sum([self.P(r, i) * self.M(r, aa) for r in amino_acids])

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
        df = pd.read_csv(osp.join("hhblits", uid), sep=',')
        self._profile = [df.loc[i].to_dict() for i in range(len(df))]

    def __getitem__(self, t):
        i, a = t
        pos_dict = self._profile[i]
        return pos_dict[a]

    def __len__(self):
        return len(self._profile)


def zscore_filter(ys, eps=10e-6, threshold=3.5):
    median_y = np.median(ys)
    median_absolute_deviation_y = np.median([np.abs(y - median_y) for y in ys]) + eps
    modified_z_scores = [0.6745 * (y - median_y) / median_absolute_deviation_y for y in ys]
    return np.asarray(ys)[(np.abs(modified_z_scores) <= threshold)]


class SkempiStruct(PDB):

    def __init__(self, pdb_struct, chains_a, chains_b, profiles=None, alignments=None, simulated=False):
        cs = list(chains_a + chains_b)
        super(SkempiStruct, self).__init__(pdb_struct.pdb,
                                           pdb_struct._chains,
                                           dict(zip(cs, cs)))
        self.profiles = profiles
        self.alignments = alignments
        self.init_profiles()
        self.init_alignments()
        self.chains_a = chains_a
        self.chains_b = chains_b
        self.stride = get_stride(self, chains_a, chains_b)
        self.simulated = simulated

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

    def get_residue(self, mut):
        return self[mut.chain_id][mut.i]

    def get_consrv(self, chain_id, posi, eps=0.0001):
        w = self[chain_id][posi].name
        if w == 'X':
            return 0.0     # zero information
        try:
            return -np.log(max(self.get_profile(chain_id)[(posi, w)], eps))
        except IndexError as e:
            raise e

    def get_ss(self, chain_id, posi):
        try: return self.stride[(chain_id, posi)]["SS"]
        except KeyError: return 'C'

    def get_acc1(self, chain_id, posi):
        try: d = self.stride[(chain_id, posi)]["ASA"]
        except KeyError: return 0.0
        w = self[chain_id][posi].name
        if w == 'X': return 1 - (float(d) / np.mean(MaxASA_emp.values()))
        return 1 - (float(d) / MaxASA_emp[w])

    def get_acc2(self, chain_id, posi):
        try: d = (lambda dic: dic["ASA_Chain"] - dic["ASA"])(self.stride[(chain_id, posi)])
        except KeyError: return 0.0
        w = self[chain_id][posi].name
        if w == 'X': return float(d) / np.mean(MaxASA_emp.values())
        return float(d) / MaxASA_emp[w]

    @property
    def protein(self):
        return "%s_%s_%s" % (self.pdb[:4], self.chains_a, self.chains_b)

    def __eq__(self, other):
        return self.protein == other.protein

    @property
    def num_chains(self):
        return len(self.chains_a + self.chains_b)

    @property
    def modelname(self):
        return self.pdb


class ProthermStruct(SkempiStruct):
    def __init__(self, pdb_struct, profiles=None, alignments=None):
        chains_a = chains_b = ''.join([c for c in pdb_struct.chains])
        super(ProthermStruct, self).__init__(pdb_struct, chains_a, chains_b, profiles, alignments)

    def get_acc2(self, chain_id, posi):
        return 0.0


def simulate_mutations(structure, mutations, simulated=True):
    mutantname, ws = apply_modeller(structure, mutations)
    pth = osp.join(ws, "%s.pdb" % mutantname)
    with open(pth, 'r') as f:
        modeled_struct = parse_pdb(mutantname, f)
    profiles, alignments = structure.profiles, structure.alignments
    ca, cb = structure.chains_a, structure.chains_b
    s = SkempiStruct(modeled_struct, ca, cb, profiles, alignments, simulated=simulated)
    return s


class SkempiRecord(object):

    def __init__(self, skempi_struct, mutations, ddg_arr=[], load_mutant=True, modeller_struct=False):
        self.ddg_arr = ddg_arr
        self.struct = skempi_struct
        self.mutant = None
        assert all([skempi_struct[m.chain_id][m.i].name == m.w for m in mutations])
        self.mutations = mutations
        if load_mutant:
            self.mutant = simulate_mutations(self.struct, self.mutations, simulated=False)
        if modeller_struct:
            self.struct = simulate_mutations(self.struct, [mut.identity for mut in self.mutations])
        self.fx4 = foldx4(self.struct, self.mutations)
        self.reverse = False

    @property
    def ddg(self):
        return np.mean(zscore_filter(self.ddg_arr))

    @property
    def pdb(self):
        return self.struct.pdb[:4].upper()

    @property
    def features(self, include_foldx4=True):
        feats = get_features(self.mutant if self.reverse else self.struct, self.mutations).values()
        return [self.fx4] + feats if include_foldx4 else feats

    def __reversed__(self):
        assert self.mutant is not None
        record = SkempiRecord(self.mutant,
                              [reversed(mut) for mut in self.mutations],
                              [-v for v in self.ddg_arr],
                              load_mutant=False, modeller_struct=False)
        record.reverse = not self.reverse
        record.mutant = self.struct
        record.fx4 = foldx4(record.struct, record.mutations)
        return record

    def __str__(self):
        muts = ','.join([str(m) for m in self.mutations])
        return "<%s: %s>" % (self.struct.protein, muts)

    def __hash__(self):
        return hash(str(self))


class ProthermRecord(SkempiRecord):

    def __init__(self, skempi_struct, mutations, ddg_arr=[], load_mutant=True):
        super(ProthermRecord, self).__init__(skempi_struct, mutations, ddg_arr, load_mutant)

    def init_mutant(self):
        wt = self.struct
        mutantname, ws = apply_modeller(wt, self.mutations)
        pth = osp.join(ws, "%s.pdb" % mutantname)
        mutant_sturct = parse_pdb(mutantname, open(pth, 'r'))
        profiles, alignments = self.struct.profiles, self.struct.alignments
        self.mutant = ProthermStruct(mutant_sturct, profiles, alignments)


class Mutation(object):

    def __init__(self, mutation_str):
        self._str = mutation_str
        self.iw, self.im = iw, im = (0, -1)
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

    @property
    def identity(self):
        mutation_str = "%s%s" % (self._str[:self.im], self._str[self.iw])
        return Mutation(mutation_str)

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
    cp_A, cp_B = get_cp_in_shell_around_res(res, struct.atoms, inner, outer, mat)
    return cp_A[mut.m], cp_B[mut.m]


def get_cps_shell(struct, mutations, inner, outer, mat=BASU010101):
    return [comp_cp_in_shell(mut, struct, mat, inner, outer) for mut in mutations]


def comp_cp_in_sphere(mut, struct, mat, rad):
    res = struct[mut.chain_id][mut.i]
    cp_A, cp_B = get_cp_in_sphere_around_res(res, struct.atoms, rad, mat)
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
    total = 0.0
    for mut in mutations:
        total += 1.0
        ret[AAA_dict[mut.w]] += 1.0
        ret[AAA_dict[mut.m]] -= 1.0
    return ret


def score_cp(struct, mut, inner, outer, mat=BASU010101):
    w, m = mut.w, mut.m
    center_res = struct[mut.chain_id][mut.i]
    neighbors = get_atoms_in_shell_around_center(center_res.ca.coord, struct.atoms, inner, outer, CNOS)
    residues_indices = set([(a.res.chain_id, a.res.index) for a in neighbors
                            if a.res.chain.chain_id != mut.chain_id])
    return sum([sum([struct.get_profile(A)[(j, a)] * (mat[(a, m)] - mat[(a, w)])
                for a in amino_acids]) for A, j in residues_indices])


def score_cp46(struct, mut, mat=BASU010101):
    return score_cp(struct, mut, 4.0, 6.0, mat=mat)


def score_cp68(struct, mut, mat=BASU010101):
    return score_cp(struct, mut, 6.0, 8.0, mat=mat)


def score_bv(struct, mut, PBV=BASU010101, rad=4.0):
    w, m = mut.w, mut.m
    center_res = struct[mut.chain_id][mut.i]
    neighbors = get_atoms_in_sphere_around_res(center_res, struct.atoms, rad)
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


def ac_ratio(st, chain_id, pos):
    try:
        d = (lambda dic: dic["ASA_Chain"] - dic["ASA"])(st.stride[(chain_id, pos)])
        w = st[chain_id][pos].name
        return float(d) / MaxASA_emp[w]
    except KeyError, e:
        raise e


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


def get_scores(score_func, struct, mutations, with_ac=True):
    if with_ac:
        return [ac_ratio(struct, m.chain_id, m.i) * score_func(struct, m) for m in mutations]
    else:
        return [score_func(struct, m) for m in mutations]


def compute_score(score_func, struct, mutations, agg=agg_multiple_scores_to_one, with_ac=True):
    return agg(get_scores(score_func, struct, mutations, with_ac=with_ac))


def get_features(struct, mutations):
    feats = dict()
    feats["Hp"] = compute_score(score_hp, struct, mutations, agg=np.mean)
    # feats["Mw"] = compute_score(score_molweight, struct, mutations, agg=np.mean)
    # feats["MaxASA"] = compute_score(score_asa, struct, mutations, agg=np.mean)
    feats["EVO"] = compute_score(score_evo, struct, mutations, agg=np.sum)
    feats["BI"] = compute_score(score_bi, struct, mutations, agg=np.sum)
    # feats["CP46"] = compute_score(score_cp46, struct, mutations, agg=np.sum)
    # feats["CP68"] = compute_score(score_cp68, struct, mutations, agg=np.sum)
    feats["BV"] = compute_score(score_bv, struct, mutations, agg=np.sum)
    feats["SK"] = compute_score(score_sk, struct, mutations, agg=np.sum)
    # feats.update(get_counts(mutations))
    return feats


def get_neural_features(struct, mutations):
    feats = dict()
    feats["Feats"] = [Descriptor(struct, mut).scores for mut in mutations]
    feats["IntAct"] = [get_interactions(struct, mut) for mut in mutations]
    feats["Pos"] = [[amino_acids.index(mut.w) + 1] for mut in mutations]
    feats["Mut"] = [[amino_acids.index(mut.m) + 1] for mut in mutations]
    feats["Prof"] = [[struct.get_profile(mut.chain_id)[(mut.i, aa)] for aa in amino_acids] for mut in mutations]
    return feats


class Descriptor(object):

    def __init__(self, struct, mut):
        self.struct = struct
        self.mut = mut

    @property
    def scores(self):
        hp = score_hp(self.struct, self.mut)
        bi = score_bi(self.struct, self.mut)
        sk = score_sk(self.struct, self.mut)
        bv = score_bv(self.struct, self.mut)
        evo = score_evo(self.struct, self.mut)
        return np.asarray([hp, bi, sk, bv, evo])


class Interaction(object):

    def __init__(self, st, mut, atom_a, atom_b, dist=None):
        self.mut = mut
        self.struct = st
        self.atom_a = atom_a
        self.atom_b = atom_b
        self._dist = dist

    @staticmethod
    def pad():
        return [0, 0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    @property
    def dist(self):
        if self._dist is None:
            a, b = self.atom_a, self.atom_b
            self._dist = math.sqrt(sum([(x1-x2)**2 for x1, x2 in zip(a.coord, b.coord)]))
        return self._dist

    def __eq__(self, other):
        return \
            ((self.atom_a == other.atom_a) and (self.atom_b == other.atom_b)) or \
            ((self.atom_a == other.atom_b) and (self.atom_b == other.atom_a))

    @property
    def descriptor(self):
        a, b = self.atom_a, self.atom_b
        at1, at2 = ATOM_TYPES.index(a.type) + 1, ATOM_TYPES.index(b.type) + 1
        ap1, ap2 = ATOM_POSITIONS.index(a.pos) + 1, ATOM_POSITIONS.index(b.pos) + 1
        aa1, aa2 = amino_acids.index(a.res.name) + 1, amino_acids.index(b.res.name) + 1
        pr_mut = self.struct.get_profile(self.mut.chain_id)[(self.mut.i, self.mut.m)]
        pr1 = self.struct.get_profile(a.chain_id)[(a.res.index, b.res.name)]
        pr2 = self.struct.get_profile(b.chain_id)[(b.res.index, a.res.name)]
        ac11 = self.struct.get_acc1(a.chain_id, a.res.index)
        ac12 = self.struct.get_acc1(b.chain_id, b.res.index)
        ac21 = self.struct.get_acc2(a.chain_id, a.res.index)
        ac22 = self.struct.get_acc2(b.chain_id, b.res.index)
        return [amino_acids.index(self.mut.m) + 1, aa1, aa2, at1, at2, ap1, ap2,
                self.dist, pr1, pr2, pr_mut, ac11, ac12, ac21, ac22]

    def __str__(self):
        a, b = self.atom_a, self.atom_b
        return "%s:%s--(%.3f)--%s:%s" % (a.res._name, a.name, self.dist, b.name, b.res._name)


def get_interactions(struct, mut, rad=4.0, ignore_list=BACKBONE_ATOMS):
    from sklearn.metrics.pairwise import euclidean_distances as dist
    center_res = struct[mut.chain_id][mut.i]
    assert center_res.name == mut.w or center_res.name == mut.m
    X = [a.coord for a in center_res.atoms]
    residues = list(set([a.res for a in get_atoms_in_sphere_around_res(center_res, struct.atoms, rad)]))
    envs = [dist(X, [a.coord for a in rr.atoms]) for rr in residues]
    indices = [np.unravel_index(np.argmin(e, axis=None), e.shape) for e in envs]
    interactions = [Interaction(struct, mut, center_res[i], residues[k][j], envs[k][i, j]) for k, (i, j) in enumerate(indices)]
    filtered_interactions = sorted([ii for ii in interactions if ii.atom_a.name not in ignore_list], key=lambda x: x.dist)
    return filtered_interactions


class Dataset(object):

    def __init__(self, list_of_records):
        self._records = list_of_records
        self._X = np.asarray([rec.features for rec in list_of_records], dtype=np.float64)
        self.df = pd.DataFrame([rec.ddg for rec in list_of_records], columns=["DDG"])
        self.df["Mutation"] = [','.join([str(m) for m in rr.mutations]) for rr in list_of_records]
        self.df["Protein"] = [rr.struct.protein for rr in list_of_records]
        self.df["Num_Muts"] = [len(rr.mutations) for rr in list_of_records]
        self.df["Num_Chains"] = [rr.struct.num_chains for rr in list_of_records]

    def __reversed__(self):
        return Dataset([reversed(rec) for rec in self._records])

    @property
    def shape(self):
        return self._X.shape

    @property
    def X(self):
        return np.copy(self._X)

    @property
    def fx4(self):
        return np.copy(self._X[:, 0])

    @property
    def y(self):
        return self.df.DDG.values

    @property
    def proteins(self):
        return self.df.Protein

    @property
    def mutations(self):
        return self.df.Mutation

    @property
    def num_chains(self):
        return self.df.Num_Chains

    @property
    def num_muts(self):
        return self.df.Num_Muts

    def __getitem__(self, i):
        return self._records[i]

    def __len__(self):
        return len(self._records)


def protherm_generator(dataframe, path_to_pdbs, load_mut=True, min_seq_length=0):
    structs, records = {}, {}
    for i in tqdm(range(len(dataframe)), "loading structures"):
        row = dataframe.loc[i]
        if row.PDB in structs:
            st = structs[row.PDB]
        else:
            pth = osp.join(path_to_pdbs, "%s.pdb" % row.PDB.lower())
            st = structs[row.PDB] = parse_pdb(row.PDB, open(pth, 'r'))
        if np.any([len(c) <= min_seq_length for c in st._chains]):
            continue
        key = (row.PDB, row.Mut)
        if key not in records:
            records[key] = [st]
        records[key].append(row.DDG)
    prepare_alignment_profiles(structs.values())
    structs = {}
    for (_, ms), v in tqdm(records.items(), "loading records"):
        key, ddg_arr = v[0], v[1:]
        if key not in structs:
            structs[key] = ProthermStruct(key)
        st, ddg_arr = structs[key], v[1:]
        yield ProthermRecord(st, parse_mutations(ms), ddg_arr, load_mutant=load_mut)


def load_protherm(dataframe, path_to_pdbs, load_mut=True, min_seq_length=0):
    return list(protherm_generator(dataframe, path_to_pdbs, load_mut, min_seq_length))


def prepare_alignment_profiles(structs):
    seqs = []
    for st in structs:
        seqs.extend([(c.id, c.seq) for c in st._chains if not osp.exists('hhblits/%s.aln' % c.id)])
    run_hhblits(seqs)
    seqs = []
    for st in structs:
        seqs.extend([(c.id, c.seq) for c in st._chains if not osp.exists('hhblits/%s.prof' % c.id)])
    comp_profiles(seqs)


def load_varibench(): return load_protherm(varib_df, PROTHERM_PDBs, True, 0)


def load_s2648(): return load_protherm(s2648_df, PROTHERM_PDBs, True, 0)


def skempi_generator(dataframe, path_to_pdbs, load_mut=True, modeller_struct=False, min_seq_length=0):
    structs, records = {}, {}
    for i in tqdm(range(len(dataframe)), "loading structures"):
        row = dataframe.loc[i]
        t = tuple(row.Protein.split('_'))
        if t in structs:
            st = structs[t]
        else:
            pth = osp.join(path_to_pdbs, "%s.pdb" % t[0])
            st = structs[t] = parse_pdb(t[0], open(pth, 'r'))
        if np.any([len(c) <= min_seq_length for c in st._chains]):
            continue
        key = (t, row["Mutation(s)_cleaned"])
        if key not in records:
            records[key] = [st]
        records[key].append(row.DDG)
    prepare_alignment_profiles(structs.values())
    structs = {}
    for k, v in tqdm(records.items(), "loading records"):
        (_, ca, cb), ms = k
        key = (v[0], ca, cb)
        if key not in structs:
            structs[key] = SkempiStruct(v[0], ca, cb)
        st, ddg_arr = structs[key], v[1:]
        yield SkempiRecord(st, parse_mutations(ms), ddg_arr, load_mutant=load_mut, modeller_struct=modeller_struct)


def load_skempi(dataframe, path_to_pdbs, load_mut=True, modeller_struct=False, min_seq_length=0):
    return list(skempi_generator(dataframe, path_to_pdbs, load_mut, modeller_struct, min_seq_length))


def prepare_skempi_v2(path_to_csv=osp.join('..', 'data', 'skempi_v2.csv')):
    with open(path_to_csv, 'r') as f:
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
    df.Temperature = df.Temperature.replace(r'\(assumed\)', '', regex=True).replace(r'', '298', regex=False)
    indx = (df.Temperature != '') & (df["Affinity_mut_parsed"] != '') & (df["Affinity_wt_parsed"] != '')
    df = df[indx].reset_index()
    df["Temperature"] = T = df.Temperature.astype(float).values
    df["num_mutations"] = np.asarray(num_mutations)[indx]
    df["Protein"] = df["#Pdb"]
    df["version"] = df["SKEMPI version\n"].astype(int)
    R = (8.314 / 4184)
    df["dGwt"] = R * T * np.log(df["Affinity_wt_parsed"].astype(float).values)
    df["dGmut"] = R * T * np.log(df["Affinity_mut_parsed"].astype(float).values)
    df["DDG"] = df["dGmut"] - df["dGwt"]
    df.to_csv(osp.join('..', 'data', 'skempi_v2.1.csv'), index=False)
    return df.reset_index(drop=True)


def prepare_varibench(basedir="../data/mutation_data_sets/crossval_varibench"):
    return prepare_crossval_dataset(basedir)


def prepare_s2648(basedir="../data/mutation_data_sets/crossval_s2648"):
    return prepare_crossval_dataset(basedir)


def prepare_crossval_dataset(basedir, lim=None):
    data = []
    for fname in os.listdir(basedir)[:lim]:
        ll = [l.strip().split() for l in open(osp.join(basedir, fname), 'r').readlines()]
        ll = [(r[0].upper()[:4], r[1][:1]+r[0][4]+r[1][1:], -float(r[2])) for r in ll]
        data.extend(ll)
    return pd.DataFrame(data, columns=["PDB", "Mut", "DDG"]).drop_duplicates().reset_index(drop=True)


try:
    skempi_df = pd.read_excel(osp.join('..', 'data', 'SKEMPI_1.1.xlsx'))
    skempi_df["num_chains"] = skempi_df.Protein.str.slice(start=6).apply(len).values
    skempi_df["num_muts"] = skempi_df['Mutation(s)_cleaned'].str.split(',').apply(len).values
    skempi_df_v2 = prepare_skempi_v2()
    skempi_df_v2["num_chains"] = skempi_df_v2.Protein.str.slice(start=6).apply(len).values
    skempi_df_v2["num_muts"] = skempi_df_v2['Mutation(s)_cleaned'].str.split(',').apply(len).values
    varib_df = prepare_varibench()
    s2648_df = prepare_s2648()

except IOError as e:
    print("warning: %s" % e)
    skempi_df = None
    skempi_df_v2 = None


if __name__ == "__main__":
    s_v1 = set(skempi_df_v2[skempi_df_v2.version == 1].Protein)
    ix_v1 = (skempi_df_v2.version == 1) | skempi_df_v2.Protein.isin(s_v1)
    ix_v2 = (skempi_df_v2.version == 2) & ~skempi_df_v2.Protein.isin(s_v1)
    records_v1 = load_skempi(skempi_df_v2[ix_v1].reset_index(drop=True), SKMEPI2_PDBs, True, True)
    dataset_v1 = Dataset(records_v1)
    records_v2 = load_skempi(skempi_df_v2[ix_v2].reset_index(drop=True), SKMEPI2_PDBs, True, True)
    dataset_v2 = Dataset(records_v2)
