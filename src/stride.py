import os
import sys
import subprocess
import numpy as np
import pandas as pd


STRIDE_EXE = '../stride/stride'


def delta_sasa(chainA, chainB, path_to_pdb):
    args = [STRIDE_EXE, path_to_pdb, '-r%s%s' % (chainA, chainB)]
    proc0 = subprocess.Popen(args, stdout=subprocess.PIPE)
    df_complex = parse_stride(proc0.stdout)
    args = [STRIDE_EXE, path_to_pdb, '-r%s' % (chainA,)]
    proc1 = subprocess.Popen(args, stdout=subprocess.PIPE)
    try: df_a = parse_stride(proc1.stdout)
    except ValueError: df_a = df_complex[df_complex.Chain == chainA]
    args = [STRIDE_EXE, path_to_pdb, '-r%s' % (chainB,)]
    proc2 = subprocess.Popen(args, stdout=subprocess.PIPE)
    try: df_b = parse_stride(proc2.stdout)
    except ValueError: df_b = df_complex[df_complex.Chain == chainB]
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


def main(*args, **kwargs):
    if not os.path.exists('./stride'):
        os.makedirs('./stride')
    pdb_path, chainA, chainB, out_path  = args
    delta_sasa(chainA, chainB, pdb_path).to_csv(out_path, index=False)


if __name__ == "__main__":
    pdb_path, chainA, chainB, out_path = sys.argv[1:]
    main(out_path, chainA, chainB, pdb_path)
