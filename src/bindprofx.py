import os
import sys
import shutil
import pandas as pd

from tqdm import tqdm
from scipy.stats import pearsonr
from itertools import combinations as comb
from skempi_consts import *

BINDPROFX_HOME = "../../BindProfX/bin"

BINDPROFX_DATA = "../data/XBindProfPaperData"

BPX_PDB_PATH = "../data/XBindProfPaperData/pdbs"


try:
    bpx_paper_df = pd.read_excel(osp.join(BINDPROFX_DATA, 'all.xlsx'))
except IOError as e:
    print("warning: %s" % e)
    bpx_paper_df = None


class MutList(object):

    def __init__(self, mutations):
        self.mutations = mutations

    def __str__(self):
        return "%s;\n" % ",".join([str(m) for m in self.mutations])


class Alignment(object):

    def __init__(self, path, threshold=0.5):
        with open(path, "r") as fd:
            lines = fd.readlines()
        self.header = lines[:6]
        assert lines[5] == "Alignments:\n"
        self.body = [lines[6]]
        i = 1
        while float(lines[6 + i].split()[5]) >= threshold:
            self.body.append(lines[6 + i])
            i += 1

    def to_file(self, path):
        with open(path, "w+") as f:
            f.writelines(self.header + self.body)


def get_bpx_score(result_path):
    with open(result_path) as f:
        score = float(f.read().strip().split(' ')[0])
    return score


def bindprofx(skempi_record, bindprofx_home=BINDPROFX_HOME, bindprofx_data=BINDPROFX_DATA):
    struct = skempi_record.struct
    mutations = skempi_record.mutations
    mutlist = MutList(mutations)
    ws = "bindprofx/%s" % struct.modelname
    result_path = "%s/result.txt" % ws
    # if osp.exists(result_path):
    #     return get_bpx_score(result_path)
    if not osp.exists(ws):
        os.makedirs(ws)
    src = "%s/align/%s.aln" % (bindprofx_data, struct.pdb)
    dst = "%s/align.out" % ws
    shutil.copy(src, dst)
    Alignment(dst).to_file(dst)
    with open("%s/mutList.txt" % ws, "w+") as f:
        f.write(str(mutlist))
    struct.struct.to_pdb("%s/complex.pdb" % ws)
    cline = "%s %s/get_final_score.py %s" % (sys.executable, bindprofx_home, ws)
    assert os.WEXITSTATUS(os.system(cline)) == 0
    return get_bpx_score(result_path)


def bindprofx_predict(skempi_records, evlaute_negative=False):
    data = []
    for r in tqdm(skempi_records, desc="records processed"):
        if r.group == 0:
            continue
        if evlaute_negative:
            r = reversed(r)
        mutations = ",".join([str(m) for m in r.mutations])
        data.append([r.modelname, mutations, r.group, r.ddg, bindprofx(r)])
        colnames = ["MODEL_NAME", "MUTATIONS", "GROUP", "DDG", "BINDPROFX"]
    return pd.DataFrame(data, columns=colnames)


def bindprofx_evaluate(df_bpx):
    data = []
    for i, (g1, g2) in enumerate(comb(range(1, NUM_GROUPS+1), 2)):
        df = df_bpx[df_bpx.GROUP.isin((g1,g2))]
        true = df.DDG
        preds = df.BINDPROFX
        cor, _ = pearsonr(true, preds)
        data.append(["G%d" % g1, "G%d" % g2, cor])
    return pd.DataFrame(data, columns=["Group1", "Group2", "PCC"])


if __name__ == "__main__":
    pass
