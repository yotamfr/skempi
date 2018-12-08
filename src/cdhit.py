import os
import json
import itertools
import numpy as np
from pdb_utils import to_fasta

CDHIT_HOME = "/media/disk1/yotam/skempi/cdhit"


def get_cdhit_clusters(fasta_filename,
                       parse=lambda seq: seq.split('>')[1].split('...')[0].split('|')[0]):
    cline = '%s/cd-hit -i %s -o %s_60 -c 0.6 -n 4' % (CDHIT_HOME, fasta_filename, fasta_filename)
    assert os.WEXITSTATUS(os.system(cline)) == 0
    cline = '%s/psi-cd-hit/psi-cd-hit.pl -i %s -o %s_30 -c 0.3' % (CDHIT_HOME, fasta_filename, fasta_filename)
    assert os.WEXITSTATUS(os.system(cline)) == 0
    cline = '%s/clstr_rev.pl %s_60.clstr %s_30.clstr > %s_60-30.clstr' % (
    CDHIT_HOME, fasta_filename, fasta_filename, fasta_filename)
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


def similar(st1, st2, chain_to_cluster, min_len=15):
    for c1, c2 in itertools.product(st1.chains.values(), st2.chains.values()):
        if (len(c1) <= min_len) or (len(c2) <= min_len):
            continue
        if chain_to_cluster[c1.id] != chain_to_cluster[c2.id]:  # found 2 chains with identity >= 30%
            return False
    return True


# def verify_dissimilarity(list_of_structs, chain_to_cluster, struct_to_cluster):
#     for cl in np.random.permutation(struct_to_cluster.values()):
#         inside = [st for st in list_of_structs if struct_to_cluster[st] == cl]
#         outside = [st for st in list_of_structs if struct_to_cluster[st] != cl]
#         for inn, out in itertools.product(inside, outside):
#             if similar(inn, out, chain_to_cluster):
#                 return struct_to_cluster[inn], struct_to_cluster[out]
#     return None


# def size_of(g, struct_to_cluster):
#     return len({st: cl for st, cl in struct_to_cluster.items() if cl == g})


# def join(cl1, cl2, struct_to_cluster):
#     sz1 = size_of(cl1, struct_to_cluster)
#     sz2 = size_of(cl2, struct_to_cluster)
#     if sz1 < sz2:
#         for st in struct_to_cluster.keys():
#             if struct_to_cluster[st] == cl2:
#                 struct_to_cluster[st] = cl1
#     else:
#         for st in struct_to_cluster.keys():
#             if struct_to_cluster[st] == cl1:
#                 struct_to_cluster[st] = cl2


# def divide_structs_into_groups(list_of_structs, fasta_filename, num_groups=10):
#     to_fasta(list_of_structs, fasta_filename)
#     _, chain_to_cluster = get_cdhit_clusters(fasta_filename)
#     struct_to_cluster = {st: i for i, st in enumerate(list_of_structs)}
#     while len(set(struct_to_cluster.values())) > num_groups:
#         print(len(set(struct_to_cluster.values())))
#         clusters = set(struct_to_cluster.values())
#         cls = sorted([[cl, size_of(cl, struct_to_cluster)] for cl in clusters], key=lambda p: p[1])
#         cl1, cl2 = cls[0][0], cls[1][0]
#         join(cl1, cl2, struct_to_cluster)
#         culprit = verify_dissimilarity(list_of_structs, chain_to_cluster, struct_to_cluster)
#         while culprit is not None:
#             cl1, cl2 = culprit
#             join(cl1, cl2, struct_to_cluster)
#             culprit = verify_dissimilarity(list_of_structs, chain_to_cluster, struct_to_cluster)
#     return struct_to_cluster


def are_similar(struct, outside_structs, chain_to_cluster):
    for st1, st2 in itertools.product([struct], outside_structs):
        if similar(st1, st2, chain_to_cluster):
            return True
    return False


def divide_structs_into_existing_groups(list_of_structs, groups, fasta_filename="../data/skempi_v2"):
    to_fasta(list_of_structs, fasta_filename)
    _, chain_to_cluster = get_cdhit_clusters(fasta_filename)
    remainder = []
    list_of_structs = [o for o in list_of_structs if not np.any([o in gg for gg in groups])]
    prot_to_struct = {ss.protein: ss for ss in list_of_structs}
    while list_of_structs:
        sim = True
        s = list_of_structs.pop()
        for ig in sorted(range(len(groups)), key=lambda i: len(groups[i])):
            grp = groups[ig]
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
    records1 = load_skempi(skempi_df, PDB_PATH, False, 0)
    records2 = load_skempi(skempi_df_v2, SKMEPI2_PDBs, False, 0)
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
