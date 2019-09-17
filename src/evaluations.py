from skempi_consts import *

from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from itertools import combinations as comb

from math import sqrt

import pandas as pd
import numpy as np

from scipy.stats import pearsonr

mse = mean_squared_error

rmse = lambda(y, y_hat): sqrt(mean_squared_error(y, y_hat))


def get_stats(x1, x2, lbl1="V1", lbl2="V2", eps=0.1):
    return pd.DataFrame([
        ["|X|", int(len(x1)), int(len(x2))],
        ["E(X)", np.mean(x1), np.mean(x2)],
        ["Min", np.min(x1), np.min(x2)],
        ["Max", np.max(x1), np.max(x2)],
        ["Var(X)", np.var(x1), np.var(x2)],
        ["SD", np.std(x1), np.std(x2)],
        ["Median", np.median(x1), np.median(x2)],
        ["Pr(|x|<=1)", np.sum(np.abs(x1) <= 1) / float(len(x1)), np.sum(np.abs(x2) <= 1) / float(len(x2))]],
        columns=["Stat", lbl1, lbl2])


def print_cv_stats(pcc_trn, pcc_tst, mse_trn, mse_tst):
    print("PCC: %.3f, %.3f" % (pcc_trn, pcc_tst))
    print("MSE: %.3f, %.3f" % (mse_trn, mse_tst))


def run_zhang_cv_test(dataset, get_regressor, save_prefix=None, normalize=0, groups=DIMER_GROUPS, h=2):
    results = {"Group1": [], "Group2": [],
               "PCC_TRN": [], "PCC_TST": [],
               "MSE_TRN": [], "MSE_TST": []}
    prots = []
    for g in groups: prots.extend(g)
    X, y, names = dataset.X, dataset.y, dataset.proteins
    for i, pair in enumerate(comb(range(len(groups)), 2)):
        group = groups[pair[0]] + groups[pair[1]]
        rest = list(set(prots) - set(group))
        indx_tst = names.isin(group)
        indx_trn = names.isin(rest)
        y_trn = y[indx_trn]
        y_tst = y[indx_tst]
        X_trn = X[indx_trn]
        X_tst = X[indx_tst]
        regressor = get_regressor(pair[0], pair[1])
        if normalize == 1:
            scaler = StandardScaler()
            scaler.fit(X_trn)
            X_trn = scaler.transform(X_trn)
            X_tst = scaler.transform(X_tst)
        if save_prefix is not None:
            g1, g2 = pair[0] + 1, pair[1] + 1
            print("training for G%d, G%d..." % (g1, g2))
            regressor.fit(X_trn, y_trn, valid=[X_tst, y_tst], prefix="G%dG%d" % (g1, g2))
            pth = 'models/%s%d%d.pkl' % (save_prefix, pair[0], pair[1])
            joblib.dump(regressor, pth)
        y_hat_trn = regressor.predict(X_trn)
        cor_trn, _ = pearsonr(y_trn, y_hat_trn)
        y_hat_tst = regressor.predict(X_tst)
        cor_tst, _ = pearsonr(y_tst, y_hat_tst)
        mse_trn = mse(y_trn, y_hat_trn)
        mse_tst = mse(y_tst, y_hat_tst)
        df = pd.DataFrame([["PCC", cor_trn, cor_tst],
                          ["MSE", mse_trn, mse_tst]],
                          columns=["Stat", "TRN", "TST"]).append(
            get_stats(y_trn, y_tst, lbl1="TRN", lbl2="TST"))
        print(df.head(h))
        results["PCC_TST"].append(cor_tst)
        results["PCC_TRN"].append(cor_trn)
        results["MSE_TST"].append(mse_tst)
        results["MSE_TRN"].append(mse_trn)
        results["Group1"].append(pair[0] + 1)
        results["Group2"].append(pair[1] + 1)
    df = pd.DataFrame(results)
    print_cv_stats(np.mean(df.PCC_TRN), np.mean(df.PCC_TST), np.mean(df.MSE_TRN), np.mean(df.MSE_TST))
    return df


def run_cv_test(dataset, get_regressor, save_prefix=None, normalize=0, groups=DIMER_GROUPS, replace_nan_values=True):
    results = {"Protein": [], "Mutations": [], "Group": [], "DDG": [], "DDG_PRED": []}
    prots = []
    for g in groups:
        prots.extend(g)
    X, y, names, mutations = dataset.X, dataset.y, dataset.proteins, dataset.mutations
    if replace_nan_values:
        X[np.isnan(X)] = 0.0
    indexes = []
    for i, group in enumerate(groups):
        rest = list(set(prots) - set(group))
        indx_tst = names.isin(group)
        indx_trn = names.isin(rest)
        indexes.append([indx_trn, indx_tst])
    indx_trn = names.isin(prots)
    indx_tst = np.logical_not(indx_trn)
    indexes.append([indx_trn, indx_tst])
    for i, (indx_trn, indx_tst) in enumerate(indexes):
        y_trn = y[indx_trn]
        y_tst = y[indx_tst]
        X_trn = X[indx_trn]
        X_tst = X[indx_tst]
        regressor = get_regressor(i, i)
        if normalize == 1:
            scaler = StandardScaler()
            scaler.fit(X_trn)
            X_trn = scaler.transform(X_trn)
            X_tst = scaler.transform(X_tst)
        if save_prefix is not None:
            print("training for G%d..." % (i+1,))
            regressor.fit(X_trn, y_trn, valid=[X_tst, y_tst], prefix="G%d" % (i+1,))
            joblib.dump(regressor, 'models/%s%d%d.pkl' % (save_prefix, i, i))
        y_hat_trn = regressor.predict(X_trn)
        cor_trn, _ = pearsonr(y_trn, y_hat_trn)
        y_hat_tst = regressor.predict(X_tst)
        cor_tst, _ = pearsonr(y_tst, y_hat_tst)
        try: mse_trn = mse(y_trn, y_hat_trn)
        except ValueError: mse_trn = np.nan
        try: mse_tst = mse(y_tst, y_hat_tst)
        except ValueError: mse_tst = np.nan
        print_cv_stats(cor_trn, cor_tst, mse_trn, mse_tst)
        results["Protein"].extend(names[indx_tst])
        results["Mutations"].extend(mutations[indx_tst])
        results["DDG_PRED"].extend(y_hat_tst)
        results["DDG"].extend(y_tst)
        results["Group"].extend([i + 1] * len(y_tst))
    return pd.DataFrame(results)


def run_cross_dataset_test(dataset1, dataset2, model, name, replace_nan_values=True):
    X1, X2 = dataset1.X, dataset2.X
    if replace_nan_values:
        X1[np.isnan(X1)] = 0.0
        X2[np.isnan(X2)] = 0.0
    X_trn = X1[dataset1.num_chains <= 2]
    y_trn = dataset1.y[dataset1.num_chains <= 2]
    X_tst = X2[dataset2.num_chains <= 2]
    y_tst = dataset2.y[dataset2.num_chains <= 2]
    model.fit(X_trn, y_trn, valid=[X_tst, y_tst], prefix=name)
    y_hat_tst = model.predict(X_tst)
    y_hat_trn = model.predict(X_trn)
    cor_trn, _ = pearsonr(y_trn, y_hat_trn)
    cor_tst, _ = pearsonr(y_tst, y_hat_tst)
    mse_trn = mse(y_trn, y_hat_trn)
    mse_tst = mse(y_tst, y_hat_tst)
    print_cv_stats(cor_trn, cor_tst, mse_trn, mse_tst)
    return cor_trn, cor_tst, mse_trn, mse_tst


def plot_bar_charts_with_confidence_interval(dataframes, titles):
    import numpy as np
    import matplotlib.pyplot as plt

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * height,
                    '%.2f' % height, ha='center', va='bottom')

    fig, axarr = plt.subplots(1, len(dataframes))
    for i in range(len(dataframes)):
        ax = axarr[i]
        df = dataframes[i]
        reasons = df.columns.tolist()
        N = len(reasons)
        ind = np.arange(N)  # the x locations for the groups
        width = 0.35  # the width of the bars

        means = df.mean()
        stds = df.std()

        ax.figure.set_size_inches(4 * len(dataframes), 4)
        rects = ax.bar(ind, means, width, color='r',
                       yerr=stds if len(stds) > 0 else None)

        # add some text for labels, title and axes ticks
        ax.set_ylabel(titles[i])
        ax.set_xticks(ind)
        ax.set_xticklabels(reasons, rotation=0)

        # ax.legend(rects, [titles[i]])
        autolabel(rects)

    plt.tight_layout()


def evaluate_bfx_cv(bfx, ddg, names, groups=DIMER_GROUPS):
    results = {"Group1": [], "Group2": [], "PCC": [], "MSE": []}
    prots = []
    for G in groups: prots.extend(G)
    print("Evaluating on %d proteins" % len(prots))
    for i, pair in enumerate(comb(range(len(groups)), 2)):
        group = groups[pair[0]] + groups[pair[1]]
        indx = names.isin(group)
        y_pred = bfx[indx]
        y_true = ddg[indx]
        cor, _ = pearsonr(y_true, y_pred)
        rms = mean_squared_error(y_true, y_pred)
        print("G%d" % (pair[0]+1), "G%d" % (pair[1]+1), "%.3f" % cor)
        results["PCC"].append(cor)
        results["MSE"].append(rms)
        results["Group1"].append(pair[0]+1)
        results["Group2"].append(pair[1]+1)
    return pd.DataFrame(results)


def evaluate_bfx_holdout(bfx, ddg, names, groups=DIMER_GROUPS):
    prots = []
    for G in groups: prots.extend(G)
    print("Evaluating on %d proteins" % len(prots))
    indx = ~(names.isin(prots))
    y_pred = bfx[indx]
    y_true = ddg[indx]
    cor_tst, _ = pearsonr(y_true, y_pred)
    print("holdout", "%.3f" % cor_tst)
    return y_true, y_pred, cor_tst
