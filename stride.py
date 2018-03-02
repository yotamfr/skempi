import os
import sys
import subprocess

import numpy as np
import pandas as pd


def delta_sasa(pdb, chainA, chainB):
    args = ['stride/stride', 'data/pdbs/%s.pdb' % pdb, '-r%s%s' % (chainA, chainB)]
    proc0 = subprocess.Popen(args, stdout=subprocess.PIPE)
    df_complex = parse_stride(proc0.stdout)
    args = ['stride/stride', 'data/pdbs/%s.pdb' % pdb, '-r%s' % (chainA,)]
    proc1 = subprocess.Popen(args, stdout=subprocess.PIPE)
    df_A = parse_stride(proc1.stdout)
    args = ['stride/stride', 'data/pdbs/%s.pdb' % pdb, '-r%s' % (chainB,)]
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
            _, aa, chain, res, _, ss, _, phi, psi, asa = line.replace('~', '').split()
            info.append([aa, chain, res, ss, phi, psi, asa])
        line = out.readline()

    aas, chains, ress, sss, phis, psis, asas = zip(*info)
    return pd.DataFrame({"AA": aas, "Chain": chains, "Res": ress, "SS": sss, "Phi": phis, "Psi": psis, "ASA": asas})


if __name__ == "__main__":
    if not os.path.exists('data/stride'):
        os.makedirs('data/stride')
    pdb, chainA, chainB = sys.argv[1], sys.argv[2], sys.argv[3]
    delta_sasa(pdb, chainA, chainB).to_csv("data/stride/%s.out" % pdb, index=False)
