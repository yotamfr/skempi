import os
import time
import os.path as osp
import sys
import subprocess
import numpy as np
import pandas as pd
from skempi_consts import *


STRIDE_EXE = '../stride/stride'


class Stride(object):
    def __init__(self, stride_df, pdb_struct):
        self.dic = {}
        for i, row in stride_df.iterrows():
            d_row = row.to_dict()
            chain_id = d_row["Chain"]
            try:
                res_i = pdb_struct[chain_id].index[d_row["Res"]]
                self.dic[(chain_id, res_i)] = d_row
            except KeyError:
                pass

    def __getitem__(self, chain_res_pair):
        try:
            return self.dic[chain_res_pair]
        except KeyError, e:
            raise e


def delta_sasa(chainA, chainB, path_to_pdb):
    args = [STRIDE_EXE, path_to_pdb, '-r%s%s' % (chainA, chainB)]
    proc0 = subprocess.Popen(args, stdout=subprocess.PIPE)
    df_complex = parse_stride(proc0.stdout)
    if chainA == chainB: return df_complex
    args = [STRIDE_EXE, path_to_pdb, '-r%s' % (chainA,)]
    proc1 = subprocess.Popen(args, stdout=subprocess.PIPE)
    try: df_a = parse_stride(proc1.stdout)
    except ValueError:
        df_a = df_complex[df_complex.Chain == chainA]
    args = [STRIDE_EXE, path_to_pdb, '-r%s' % (chainB,)]
    proc2 = subprocess.Popen(args, stdout=subprocess.PIPE)
    try: df_b = parse_stride(proc2.stdout)
    except ValueError:
        df_b = df_complex[df_complex.Chain == chainB]
    ress_a, ress_b = list(df_a.Res), list(df_b.Res)
    ress = np.asarray(sorted(df_complex.Res))
    asa_a, asa_b = list(df_a.ASA), list(df_b.ASA)
    asa = np.asarray(list(df_complex.ASA))
    assert np.all(ress == np.asarray(sorted(ress_a + ress_b)))
    # assert np.any(np.asarray(asa_a + asa_b) != asa)
    df_complex["ASA_Chain"] = np.asarray(asa_a + asa_b)
    return df_complex


def parse_stride(out):
    info = list()
    line = out.readline()

    while line:
        typ = line.split()[0]
        if typ == 'ASG':
            _, aa, chain, res, _, ss, _, phi, psi, asa, _ = line.split()
            info.append([aa, chain, res, ss, phi, psi, asa])
        line = out.readline()

    aas, chains, ress, sss, phis, psis, asas = zip(*info)

    return pd.DataFrame({"AA": aas, "Chain": chains, "Res": ress,
                         "SS": sss, "Phi": phis, "Psi": psis, "ASA": asas})


def get_stride(pdb_struct, ca, cb, expiration_time_sec=EXPIRATION_TIME_SECONDS):
    modelname = pdb_struct.pdb
    pdb = modelname[:4]
    pdb_pth = osp.join('stride', modelname, '%s_%s_%s.pdb' % (pdb, ca, cb))
    out_pth = osp.join('stride', modelname, '%s_%s_%s.out' % (pdb, ca, cb))
    if not osp.exists(osp.dirname(pdb_pth)):
        os.makedirs(osp.dirname(pdb_pth))
    if not osp.exists(out_pth) or (time.time() - osp.getmtime(out_pth) > expiration_time_sec):
        pdb_struct.to_pdb(pdb_pth)
        main(pdb_pth, ca, cb, out_pth)
    return Stride(pd.read_csv(out_pth), pdb_struct)


def main(*args, **kwargs):
    if not os.path.exists('./stride'):
        os.makedirs('./stride')
    pdb_path, chainA, chainB, out_path  = args
    delta_sasa(chainA, chainB, pdb_path).to_csv(out_path, index=False)


if __name__ == "__main__":
    pdb_path, chainA, chainB, out_path = sys.argv[1:]
    main(out_path, chainA, chainB, pdb_path)
