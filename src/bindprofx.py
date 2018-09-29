import os
import sys
import shutil
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
    if not osp.exists(ws):
        os.makedirs(ws)
    struct.to_pdb("%s/complex.pdb" % ws)
    result_path = "%s/result.txt" % ws
    src = "%s/align/%s.aln" % (bindprofx_data, struct.pdb)
    dst = "%s/align.out" % ws
    if not osp.exists(src):
        os.system("%s/XBindProf/run_align.pl %s/complex.pdb 0.5 %s"
                  % (bindprofx_home, ws, src))
    shutil.copy(src, dst)
    Alignment(dst).to_file(dst)
    with open("%s/mutList.txt" % ws, "w+") as f:
        f.write(str(mutlist))
    cline = "%s %s/get_final_score.py %s" % (sys.executable, bindprofx_home, ws)
    assert os.WEXITSTATUS(os.system(cline)) == 0
    return get_bpx_score(result_path)


if __name__ == "__main__":
    pass
