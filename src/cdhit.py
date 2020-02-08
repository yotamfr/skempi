import os
import json
import itertools
import numpy as np
from pdb_utils import to_fasta

CDHIT_HOME = "/media/disk1/yotam/skempi/cdhit"


def get_cdhit_clusters(fasta_filename, parse=lambda seq: seq.split('>')[1].split('...')[0].split('|')[0], cdhit=CDHIT_HOME):
    cline = '%s/cd-hit -i %s -o %s_60 -c 0.6 -n 4' % (cdhit, fasta_filename, fasta_filename)
    assert os.WEXITSTATUS(os.system(cline)) == 0
    cline = '%s/psi-cd-hit/psi-cd-hit.pl -i %s -o %s_30 -c 0.3' % (cdhit, fasta_filename, fasta_filename)
    assert os.WEXITSTATUS(os.system(cline)) == 0
    cline = '%s/clstr_rev.pl %s_60.clstr %s_30.clstr > %s_60-30.clstr' % (cdhit, fasta_filename, fasta_filename, fasta_filename)
    assert os.WEXITSTATUS(os.system(cline)) == 0
    cluster_file, cluster_dic, reverse_dic = open("%s_60-30.clstr" % fasta_filename), {}, {}
    print("Reading cluster groups...")
    cluster_groups = (x[1] for x in itertools.groupby(cluster_file, key=lambda line: line[0] == '>'))
    for cluster in cluster_groups:
        name = int(next(cluster).strip().split()[-1])
        ids = [parse(seq) for seq in next(cluster_groups)]
        cluster_dic[name] = ids
    for cluster, ids in cluster_dic.items():
        for seqid in ids:
            reverse_dic[seqid] = cluster
    print("Detected %s clusters (>%s%% similarity) groups..." % (len(cluster_dic), 30))
    return cluster_dic, reverse_dic


def similar(st1, st2, chain_to_cluster):
    if st1 == st2:
        return True
    if len(st1._chains) != len(st2._chains):
        return False
    c2c = chain_to_cluster
    chains1 = [c for c in st1._chains if c.id in c2c]
    chains2 = [c for c in st2._chains if c.id in c2c]
    if len(chains1) != len(chains2):
        return False
    cs = list(chains1)
    for ps in itertools.permutations(chains2):
        if all([c2c[a.id] == c2c[b.id] for a, b in zip(cs, ps)]):
            return True
    return False


def are_similar(struct, outside_structs, chain_to_cluster):
    return any([similar(struct, other, chain_to_cluster) for other in outside_structs])


def divide_skempi_into_train_and_test(records_v1, records_v2, fasta_filename="../data/skempi_v2"):
    structs_v1 = set([r.struct for r in records_v1])
    structs_v2 = set([r.struct for r in records_v2])
    to_fasta(list(structs_v1 | structs_v2), fasta_filename)
    _, chain_to_cluster = get_cdhit_clusters(fasta_filename)
    testset = list()
    trainset = list()
    trainset.extend(records_v1)
    trainset_structures = [rec.struct for rec in records_v1]
    for rec in records_v2:
        if are_similar(rec.struct, trainset_structures, chain_to_cluster):
            trainset_structures.append(rec.struct)
            trainset.append(rec)
        else:
            testset.append(rec)
    return trainset, testset, chain_to_cluster


def divide_structs_into_existing_groups(list_of_structs, groups, fasta_filename="../data/skempi_v2"):
    to_fasta(list_of_structs, fasta_filename)
    _, chain_to_cluster = get_cdhit_clusters(fasta_filename)
    remainder = []
    list_of_structs = [o for o in list_of_structs if not np.any([o in gg for gg in groups])]
    prot_to_struct = {ss.protein: ss for ss in list_of_structs}
    while list_of_structs:
        sim = True
        s = list_of_structs.pop()
        for ig, grp in enumerate(sorted(range(len(groups)), key=lambda i: len(groups[i]))):
            outside = [prot_to_struct[pp] for jg, gg in enumerate(groups) if jg != ig for pp in gg]
            sim = are_similar(s, outside, chain_to_cluster)
            if not sim:
                grp.append(s.protein)
                print(ig+1, len(grp), s.protein)
                break
        if sim:
            remainder.append(s)
    return groups


if __name__ == "__main__":
    from skempi_lib import *
    records1 = load_skempi(skempi_df, PDB_PATH, False, False, 0)
    records2 = load_skempi(skempi_df_v2, SKMEPI2_PDBs, False, False, 0)
    structs1 = set([r.struct for r in records1])
    structs2 = set([r.struct for r in records2])
    all_structs = list(structs1 | structs2)
    dimers = [st for st in all_structs if st.num_chains <= 2]
    dimer_groups = divide_structs_into_existing_groups(dimers, [G1, G2, G3, G4, G5])
    json.dump({int(i+1): list(set(g)) for i, g in enumerate(dimer_groups)}, open("../data/cdhit30_dimer_groups.json", "w+"))
    for gi, grp in enumerate(dimer_groups): print(gi+1, len(grp))
    complex_groups = divide_structs_into_existing_groups(all_structs, dimer_groups)
    json.dump({int(i+1): list(set(g)) for i, g in enumerate(complex_groups)}, open("../data/cdhit30_all_groups.json", "w+"))
    for gi, grp in enumerate(complex_groups): print(gi+1, len(grp))
