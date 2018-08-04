import os
import sys
import subprocess
import numpy as np
import pandas as pd

STRIDE_EXE = '../stride/stride'


def delta_sasa(modelname, chainA, chainB):
    args = [STRIDE_EXE, '%s/%s.pdb' % (pdb_path, modelname), '-r%s%s' % (chainA, chainB)]
    proc0 = subprocess.Popen(args, stdout=subprocess.PIPE)
    df_complex = parse_stride(proc0.stdout)
    args = [STRIDE_EXE, '%s/%s.pdb' % (pdb_path, modelname), '-r%s' % (chainA,)]
    proc1 = subprocess.Popen(args, stdout=subprocess.PIPE)
    df_A = parse_stride(proc1.stdout)
    args = [STRIDE_EXE, '%s/%s.pdb' % (pdb_path, modelname), '-r%s' % (chainB,)]
    proc2 = subprocess.Popen(args, stdout=subprocess.PIPE)
    df_B = parse_stride(proc2.stdout)
    assert np.all(np.array(list(df_A["Res"]) + list(df_B["Res"])) == np.array(list(df_complex["Res"])))
    assert np.any(np.array(list(df_A["ASA"]) + list(df_B["ASA"])) != np.array(list(df_complex["ASA"])))
    df_complex.loc[:, "ASA_Chain"] = np.array(list(df_A["ASA"]) + list(df_B["ASA"]))
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


if __name__ == "__main__":
    if not os.path.exists('./stride'):
        os.makedirs('./stride')
    modelname, chainA, chainB, pdb_path = sys.argv[1:]
    delta_sasa(modelname, chainA, chainB).to_csv("./stride/%s.out" % modelname, index=False)
