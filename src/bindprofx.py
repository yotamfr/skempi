import os
import os.path as osp
import sys
import shutil
import subprocess
import pandas as pd
from concurrent.futures import *

E = ThreadPoolExecutor(8)

FOLDX4_HOME = "../../BindProfX/bin/FoldX"

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
        while 6 + i < len(lines) and float(lines[6 + i].split()[5]) >= threshold:
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
    if struct.num_chains > 2:
        return None
    ws = "bindprofx/%s" % struct.modelname
    result_path = "%s/result.txt" % ws
    if osp.exists(result_path):
        return get_bpx_score(result_path)
    mutations = skempi_record.mutations
    mutlist = MutList(mutations)

    if not osp.exists(ws):
        os.makedirs(ws)
    struct.to_pdb("%s/complex.pdb" % ws)

    src = "%s/align/%s.aln" % (bindprofx_data, skempi_record.pdb)
    dst = "%s/align.out" % ws
    if not osp.exists(src):
        os.system("%s/XBindProf/run_align.pl %s/complex.pdb 0.5 %s"
                  % (bindprofx_home, ws, src))
    Alignment(src).to_file(dst)
    with open("%s/mutList.txt" % ws, "w+") as f:
        f.write(str(mutlist))
    cline = "%s %s/get_final_score.py %s" % (sys.executable, bindprofx_home, ws)
    assert os.WEXITSTATUS(os.system(cline)) == 0
    return get_bpx_score(result_path)


def foldx4(st, mutations, foldx4_home=FOLDX4_HOME, num_retries=1):
    if st.num_chains > 2:
        return None
    chains = ','.join([c for c in st.chains_a + st.chains_b])

    if not osp.exists('foldx4'):
        os.mkdir('foldx4')
        shutil.copy("%s/rotabase.txt" % foldx4_home, 'foldx4')
        shutil.copy("%s/runFoldX.py" % foldx4_home, 'foldx4')
        shutil.copy("%s/foldx" % foldx4_home, 'foldx4')

    pdb = "%s_simulated" % st.protein if st.simulated else st.protein
    muts = "%s" % "_".join([str(m) for m in mutations])
    ws = osp.abspath("foldx4/%s/%s" % (pdb, muts))
    if osp.exists("%s/score.txt" % ws):
        return get_bpx_score("%s/score.txt" % (ws,))
    if not osp.exists(ws):
        os.makedirs(ws)

    st.to_pdb("%s/complex.pdb" % (ws,))
    with open("%s/mut_list.txt" % (ws,), "w+") as f:
        f.write("%s;\n" % ";".join([str(m) for m in mutations]))

    cline = "../../runFoldX.py complex.pdb mut_list.txt %s score.txt" % (chains,)
    try:
        p = subprocess.Popen(cline.split(), cwd=ws, stdout=open(os.devnull, 'w'), stderr=subprocess.STDOUT)
        p.communicate()
    except OSError as e:
        raise e
    if p.returncode == 0:
        return get_bpx_score("%s/score.txt" % (ws,))
    elif num_retries > 0:
        shutil.rmtree(ws)
        return foldx4(st, mutations, num_retries=num_retries-1)
    else:
        return None


if __name__ == "__main__":
    pass
