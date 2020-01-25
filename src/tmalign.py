import os
import sys
import json
import subprocess
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import *
from scipy.spatial.distance import squareform

TMALIGN_EXE = "../TMalign"

E = ThreadPoolExecutor(2)


def parse_tmalign(tmalign_out):
    lines = filter(lambda ll: ll[0] == "TM-score=", [l.split(' ') for l in tmalign_out])
    assert len(lines) == 2
    return [float(ll[1]) for ll in lines]


def run_tmalign(path_to_pdb1, path_to_pdb2):
    args = [TMALIGN_EXE, path_to_pdb1, path_to_pdb2]
    p = subprocess.Popen(args, stdout=subprocess.PIPE)
    return p.stdout


def get_distance_matrix(list_of_structs, ws="./tmalign"):
    def struct_to_path(st):
        return "./%s/%s.pdb" % (ws, st.modelname)
    if not os.path.exists(ws):
        os.mkdir(ws)
    for struct in list_of_structs:
        struct.to_pdb(struct_to_path(struct))
    results = []
    for st1, st2 in itertools.combinations(list_of_structs, 2):
        fut = E.submit(run_tmalign, struct_to_path(st1), struct_to_path(st2))
        results.append({"pdb1": st1.protein, "pdb2": st2.protein, "tmscore": fut})
    for res in tqdm(results, desc="pairs processed"):
        scores = [s for s in parse_tmalign(res["tmscore"].result())]
        res["tmscore"] = 1 - np.mean(scores)
    data = []
    for res in tqdm(results, desc="pairs processed"):
        data.append({"pdb1": res["pdb1"], "pdb2": res["pdb2"], "tmscore": res["tmscore"]})
        data.append({"pdb1": res["pdb2"], "pdb2": res["pdb1"], "tmscore": res["tmscore"]})
    df = pd.DataFrame(data).pivot("pdb1", "pdb2", "tmscore").fillna(0)
    df = df[sorted(df.columns)]
    return df


if __name__ == "__main__":
    from skempi_lib import *
    from scipy.cluster.hierarchy import fcluster
    from scipy.cluster.hierarchy import linkage
    records = load_skempi(skempi_df_v2, SKMEPI2_PDBs, False, False, 12)
    structs = list(set([r.struct for r in records]))
    df = get_distance_matrix(np.asarray(structs))
    X = squareform(df.values)
    Z = linkage(X, 'average')
    max_d = 0.70  # max_d as in max_distance
    clusters = fcluster(Z, max_d, criterion='distance')
    cc = {st.protein: int(cl) for st, cl in zip(structs, clusters)}
    json.dump(cc, open("../data/struct2group_tmalign70.json", "w+"))
