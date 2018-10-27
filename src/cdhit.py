import os
import json
import itertools
import numpy as np

CDHIT_HOME = "/media/disk1/yotam/skempi/cdhit"


def to_fasta(structs, out_file):
    lines = []
    for st in structs:
        lines.extend([">%s\n%s\n" % (c.id, c.seq) for c in st.chains.values()])
    with open(out_file, "w+") as f:
        f.writelines(lines)


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


def similar(st1, st2, chain_to_cluster):
    for c1, c2 in itertools.product(st1.chains.values(), st2.chains.values()):
        if chain_to_cluster[c1.id] == chain_to_cluster[c2.id]:  # found 2 chains with identity >= 30%
            return True
    return False


def verify_dissimilarity(list_of_structs, chain_to_cluster, struct_to_cluster):
    for cl in np.random.permutation(struct_to_cluster.values()):
        inside = [st for st in list_of_structs if struct_to_cluster[st] == cl]
        outside = [st for st in list_of_structs if struct_to_cluster[st] != cl]
        for inn, out in itertools.product(inside, outside):
            if similar(inn, out, chain_to_cluster):
                return struct_to_cluster[inn], struct_to_cluster[out]
    return None


def size_of(g, struct_to_cluster):
    return len({st: cl for st, cl in struct_to_cluster.items() if cl == g})


def join(cl1, cl2, struct_to_cluster):
    sz1 = size_of(cl1, struct_to_cluster)
    sz2 = size_of(cl2, struct_to_cluster)
    if sz1 < sz2:
        for st in struct_to_cluster.keys():
            if struct_to_cluster[st] == cl2:
                struct_to_cluster[st] = cl1
    else:
        for st in struct_to_cluster.keys():
            if struct_to_cluster[st] == cl1:
                struct_to_cluster[st] = cl2


def divide_structs_into_groups(list_of_structs, fasta_filename, num_groups=10):
    to_fasta(list_of_structs, fasta_filename)
    _, chain_to_cluster = get_cdhit_clusters(fasta_filename)
    struct_to_cluster = {st: i for i, st in enumerate(list_of_structs)}
    while len(set(struct_to_cluster.values())) > num_groups:
        print(len(set(struct_to_cluster.values())))
        clusters = set(struct_to_cluster.values())
        cls = sorted([[cl, size_of(cl, struct_to_cluster)] for cl in clusters], key=lambda p: p[1])
        cl1, cl2 = cls[0][0], cls[1][0]
        join(cl1, cl2, struct_to_cluster)
        culprit = verify_dissimilarity(list_of_structs, chain_to_cluster, struct_to_cluster)
        while culprit is not None:
            cl1, cl2 = culprit
            join(cl1, cl2, struct_to_cluster)
            culprit = verify_dissimilarity(list_of_structs, chain_to_cluster, struct_to_cluster)
    return struct_to_cluster


if __name__ == "__main__":
    from skempi_lib import *
    records = load_skempi(skempi_df_v2, SKMEPI2_PDBs, False, 12)
    structs = list(set([r.struct for r in records]))
    struct_to_cluster = divide_structs_into_groups(structs, "../data/skempi_v2", 5)
    json.dump({st.protein: cl for st, cl in struct_to_cluster.items()},
              open("../data/struct2group_cdhit30.json", "w+"))
