import os
import sys
import subprocess
import numpy as np
import pandas as pd

STRIDE_EXE = '../stride/stride'


def delta_sasa(modelname, chainA, chainB, pdb_path):
    args = [STRIDE_EXE, '%s/%s.pdb' % (pdb_path, modelname), '-r%s%s' % (chainA, chainB)]
    proc0 = subprocess.Popen(args, stdout=subprocess.PIPE)
    df_complex = parse_stride(proc0.stdout)
    args = [STRIDE_EXE, '%s/%s.pdb' % (pdb_path, modelname), '-r%s' % (chainA,)]
    proc1 = subprocess.Popen(args, stdout=subprocess.PIPE)
    df_a = parse_stride(proc1.stdout)
    args = [STRIDE_EXE, '%s/%s.pdb' % (pdb_path, modelname), '-r%s' % (chainB,)]
    proc2 = subprocess.Popen(args, stdout=subprocess.PIPE)
    df_b = parse_stride(proc2.stdout)
    ress_a, ress_b = list(df_a.Res), list(df_b.Res)
    ress = np.asarray(sorted(df_complex.Res))
    asa_a, asa_b = list(df_a.ASA), list(df_b.ASA)
    asa = np.asarray(list(df_complex.ASA))
    assert np.all(ress == np.asarray(sorted(ress_a + ress_b)))
    assert np.any(np.asarray(asa_a + asa_b) != asa)
    df_complex.loc[:, "ASA_Chain"] = np.asarray(asa_a + asa_b)
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


def main(*args, **kwargs):
    if not os.path.exists('./stride'):
        os.makedirs('./stride')
    modelname, chainA, chainB, pdb_path = args
    delta_sasa(modelname, chainA, chainB, pdb_path).to_csv("./stride/%s.out" % modelname, index=False)


if __name__ == "__main__":
    modelname, chainA, chainB, pdb_path = sys.argv[1:]
    main(modelname, chainA, chainB, pdb_path)
