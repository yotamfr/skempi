from skempi_utils import *
from aaindex import *

B = BLOSUM62
C = SKOJ970101


pdb_and_chains = set([tuple(pdb_str.split('_')) for pdb_str in skempi_df.Protein.values])


def CP(mut, skempi, C, radius=6):

    i, chain_a = mut.i, mut.chain_id
    m, w = mut.m, mut.w

    def helper(P, j):
        return sum([P[(j, a)] * (C[(a, m)] - C[(a, w)]) for a in amino_acids])

    ret = 0
    for chain_b, j in skempi.get_sphere_indices(chain_a, i,radius):

        if j == i and chain_b == chain_a:
            a = skempi[chain_b][j].name
            assert a == w
            continue

        P = skempi.get_profile(chain_b)

        ret += helper(P, j)

    return ret


prots = skempi_df.Protein.values
skempi_records = {}

for t in tqdm(set([tuple(pdb_str.split('_')) for pdb_str in prots]),
              desc="skempi entries processed"):
    skempi_records[t] = SkempiRecord(*t)


# In[12]:


def comp_ei(mut, skempi_record, B, radius):
    P = skempi_record.get_profile(mut.chain_id)
    return EI(mut.m, mut.w, P, mut.i, B)


def comp_cp(mut, skempi_record, C, radius):
    return CP(mut, skempi_record, C, radius)


def get_ddg_ei_cp_arrays(M, func, radius=None):
    arr_ddg = []
    arr_obs = []
    pbar = tqdm(range(len(skempi_df)), desc="row processed")
    for i, row in skempi_df.iterrows():
        ddg = row.DDG
        arr_ddg.append(ddg)
        arr_obs_mut = []
        for mutation in row["Mutation(s)_cleaned"].split(','):
            mut = Mutation(mutation)
            t = tuple(row.Protein.split('_'))
            skempi_record = skempi_records[t]
            skempi_record.compute_dist_mat()
            obs = func(mut, skempi_record, M, radius)
            arr_obs_mut.append(obs)
        arr_obs.append(np.sum(arr_obs_mut))
        pbar.update(1)
    pbar.close()
    return arr_ddg, arr_obs


# In[13]:


from scipy.stats import pearsonr


# In[14]:


from itertools import product


def grid_search_cp(matrices=[SKOJ970101, BASU010101], radiuses=[4, 5, 6, 7, 8]):
    res_dict = {}
    for C, angs in product(matrices, radiuses):
        key = (str(C), angs)
        arr_ddg, arr_cp = get_ddg_ei_cp_arrays(C, comp_cp, angs)
        res_dict[key] = (arr_ddg, arr_cp)
        cor_cp = pearsonr(arr_ddg, arr_cp)
        print("%s: CP: %s" % (key, cor_cp,))
    return res_dict


def grid_search_ei(matrices=[BLOSUM62, SKOJ970101, BASU010101]):
    res_dict = {}
    for B in matrices:
        key = str(B)
        arr_ddg, arr_ei = get_ddg_ei_cp_arrays(B, comp_ei)
        res_dict[key] = (arr_ddg, arr_ei)
        cor_ei = pearsonr(arr_ddg, arr_ei)
        print("%s: EI: %s" % (key, cor_ei,))
    return res_dict


# In[15]:

all_features = {}


def register_eis(eis):
    for key, val in eis.iteritems():
        _, ei = val
        all_features[("EI", key)] = ei


eis = grid_search_ei()

register_eis(eis)


save_object(all_features, "../data/all_skempi_features")


# In[16]:


# cps = grid_search_cp()


# In[17]:


def comp_cp_a_b(mut, skempi_record, C, radius):
    return CP_A_B(mut, skempi_record, C, radius)


def get_ddg_cp_a_b_arrays(M, func, radius=None):
    arr_ddg = []
    arr_obs_a = []
    arr_obs_b = []
    pbar = tqdm(range(len(skempi_df)), desc="row processed")
    for i, row in skempi_df.iterrows():
        ddg = row.DDG
        arr_ddg.append(ddg)
        arr_obs_mut_a = []
        arr_obs_mut_b = []
        for mutation in row["Mutation(s)_cleaned"].split(','):
            mut = Mutation(mutation)
            t = tuple(row.Protein.split('_'))
            skempi_record = skempi_records[t]
            skempi_record.compute_dist_mat()
            obs_a, obs_b = func(mut, skempi_record, M, radius)
            arr_obs_mut_a.append(obs_a)
            arr_obs_mut_b.append(obs_b)
        arr_obs_a.append(np.sum(arr_obs_mut_a))
        arr_obs_b.append(np.sum(arr_obs_mut_b))
        pbar.update(1)
    pbar.close()
    return arr_ddg, arr_obs_a, arr_obs_b


def grid_search_cp_a_b(matrices=[SKOJ970101, BASU010101], radiuses=[4, 5, 6, 7, 8, 9, 10]):
    res_dict = {}
    for C, angs in product(matrices, radiuses):
        key = (str(C), angs)
        arr_ddg, arr_cp_a, arr_cp_b  = get_ddg_cp_a_b_arrays(C, comp_cp_a_b, angs)
        arr_cp = np.asarray(arr_cp_a) + np.asarray(arr_cp_b)
        res_dict[key] = (arr_ddg, arr_cp_a, arr_cp_b)
        cor_cp_a = pearsonr(arr_ddg, arr_cp_a)
        cor_cp_b = pearsonr(arr_ddg, arr_cp_b)
        cor_cp = pearsonr(arr_ddg, arr_cp)
        print("%s: CP_A: %s, CP_B: %s, CP %s" % (key, cor_cp_a, cor_cp_b, cor_cp))
    return res_dict


# In[18]:


def CP_A_B(mut, skempi, C, radius=6):

    i, chain_a = mut.i, mut.chain_id
    m, w = mut.m, mut.w

#     def helper(P, j):
#         return sum([P[(j, a)] * (C[(a, m)] - C[(a, w)]) for a in amino_acids])

    def helper(a, j):
        return C[(a, m)] - C[(a, w)]

    retA, retB = 0, 0
    for chain_b, j in skempi.get_sphere_indices(chain_a, i,radius):

        a = skempi[chain_b][j].name
        if j == i and chain_b == chain_a:
            assert a == w
            continue

        P = skempi.get_profile(chain_b)

        if chain_b == chain_a:
            retA += helper(a, j)

        else:
            retB += helper(a, j)

    return retA, retB


# In[19]:


cp_a_b_s_no_profile = grid_search_cp_a_b(matrices=[SKOJ970101, BASU010101], radiuses=[6, 7])


# In[20]:


def CP_A_B(mut, skempi, C, radius=6):

    i, chain_a = mut.i, mut.chain_id
    m, w = mut.m, mut.w

    def helper(P, j):
        return sum([P[(j, a)] * (C[(a, m)] - C[(a, w)]) for a in amino_acids])

#     def helper(a, j):
#         return C[(a, m)] - C[(a, w)]

    retA, retB = 0, 0
    for chain_b, j in skempi.get_sphere_indices(chain_a, i,radius):

        a = skempi[chain_b][j].name
        if j == i and chain_b == chain_a:
            assert a == w
            continue

        P = skempi.get_profile(chain_b)

        if chain_b == chain_a:
            retA += helper(P, j)

        else:
            retB += helper(P, j)

    return retA, retB


# In[21]:


cp_a_b_s_orig = grid_search_cp_a_b(matrices=[SKOJ970101, BASU010101], radiuses=[6, 7])


# In[22]:


def CP_A_B(mut, skempi, C, radius=6):

    i, chain_a = mut.i, mut.chain_id
    m, w = mut.m, mut.w

    def helper(P, j):
        return sum([0.05 * (C[(a, m)] - C[(a, w)]) for a in amino_acids])

#     def helper(a, j):
#         return C[(a, m)] - C[(a, w)]

    retA, retB = 0, 0
    for chain_b, j in skempi.get_sphere_indices(chain_a, i,radius):

        a = skempi[chain_b][j].name
        if j == i and chain_b == chain_a:
            assert a == w
            continue

        P = skempi.get_profile(chain_b)

        if chain_b == chain_a:
            retA += helper(P, j)

        else:
            retB += helper(P, j)

    return retA, retB


# In[23]:


cp_a_b_s_uniform = grid_search_cp_a_b(matrices=[SKOJ970101, BASU010101], radiuses=[6, 7])


# In[544]:


def register_cp_a_b(cp_a_b, prefix):
    for key, val in cp_a_b.iteritems():
        _, cp_a, cp_b = val
        mat, rad = key
        all_features[(prefix, "CP_A", mat, rad)] = cp_a
        all_features[(prefix, "CP_B", mat, rad)] = cp_b


# In[545]:


register_cp_a_b(cp_a_b_s_uniform, "uniform")
register_cp_a_b(cp_a_b_s_orig, "original")
register_cp_a_b(cp_a_b_s_no_profile, "no_profile")

save_object(all_features, "../data/all_skempi_features")


# In[546]:


num_muts = np.asarray([len(mut.split(",")) for mut in skempi_df["Mutation(s)_cleaned"]])
pearsonr(skempi_df.DDG, np.log(num_muts)), pearsonr(skempi_df.DDG, num_muts)


# In[547]:


all_features["#mutations"] = np.log(num_muts)


# In[717]:


def get_stride_array(func, agg=np.sum):
    arr_stride = []
    pbar = tqdm(range(len(skempi_df)), desc="row processed")
    for i, row in skempi_df.iterrows():
        arr_obs_mut = []
        for mutation in row["Mutation(s)_cleaned"].split(','):
            mut = Mutation(mutation)
            res_i, chain_id = mut.i, mut.chain_id
            t = tuple(row.Protein.split('_'))
            skempi_record = skempi_records[t]
            stride = skempi_record.stride[(chain_id, res_i)]
            skempi_record.compute_dist_mat()
            obs = func(stride)
            arr_obs_mut.append(obs)
        arr_stride.append(agg(arr_obs_mut))
        pbar.update(1)
    pbar.close()
    return arr_stride


# In[710]:


def asa_diff(stride):
    return abs(stride["ASA"] - stride["ASA_Chain"])


stride_arr = get_stride_array(asa_diff)


# In[714]:


all_features["abs(ASA-ASA_Chain)"] = stride_arr
pearsonr(skempi_df.DDG, stride_arr)


# In[718]:


DSSP = ["G", "H", "I", "T", "E", "B", "S", "C"]

from sklearn import preprocessing
lb = preprocessing.LabelBinarizer()

lb.fit(DSSP)


def get_bin_ss(stride):
    return lb.transform([stride["SS"]])[0]


# In[731]:


ss_arr = get_stride_array(get_bin_ss, agg=lambda a: np.sum(a, axis=0))


# In[732]:


print([pearsonr(skempi_df.DDG, np.asarray(ss_arr)[:, j]) for j in range(8)])


# In[733]:


all_features.keys(), len(all_features.keys())


# In[734]:


import itertools


# In[735]:


xcor_mat = np.corrcoef(np.asarray(all_features.values()))


# In[737]:


class XCor(object):

    def __init__(self, all_features):
        self.feat_name_to_indx = {key:i for i, key in enumerate(all_features.keys())}
        self.xcor_mat = np.corrcoef(np.asarray(all_features.values()))

    def __getitem__(self, t):
        feat1, feat2 = t
        i = self.feat_name_to_indx[feat1]
        j = self.feat_name_to_indx[feat2]
        return self.xcor_mat[(i, j)]


# In[738]:


xcor = XCor(all_features)


# In[739]:


def search_min_xcor(all_features, th=0.05):
    acc = set()
    for comb in itertools.combinations(all_features.keys(), 2):
        feat1, feat2 = comb
        rho = xcor[(feat1, feat2)]
        if abs(rho) < th:
            print(feat1, feat2, rho)
            acc.add(feat1)
            acc.add(feat2)
    return acc


# In[783]:


acc_feats = search_min_xcor(all_features)


# In[784]:


len(acc_feats), acc_feats


# In[797]:

acc_feats = {
    '#mutations',
    'abs(ASA-ASA_Chain)',
    ('EI', 'BLOSUM62'),
    ('EI', 'SKOJ970101'),
    ('EI', 'BASU010101'),
    ('no_profile', 'CP_A', 'SKOJ970101', 6),
    ('no_profile', 'CP_A', 'SKOJ970101', 7),
    ('no_profile', 'CP_B', 'BASU010101', 7),
    ('no_profile', 'CP_B', 'SKOJ970101', 7),
    ('uniform', 'CP_A', 'SKOJ970101', 6),
    ('uniform', 'CP_A', 'SKOJ970101', 7),
    ('uniform', 'CP_B', 'BASU010101', 6),
    ('uniform', 'CP_B', 'BASU010101', 7),
    ('uniform', 'CP_B', 'SKOJ970101', 6),
    ('uniform', 'CP_B', 'SKOJ970101', 7)}


# In[798]:

df = skempi_df
from sklearn.preprocessing import StandardScaler


def run_cv_test(X, get_regressor, normalize=0):
    gt, preds = [], []
    for group in [G1, G2, G3, G4, G5]:
        indx_tst = df.Protein.isin(group)
        indx_trn = np.logical_not(indx_tst)
        y_trn = df.DDG[indx_trn]
        y_true = df.DDG[indx_tst]
        X_trn = X[indx_trn]
        X_tst = X[indx_tst]
        regressor = get_regressor()
        if normalize == 1:
            scaler = StandardScaler()
            scaler.fit(X_trn)
            X_trn, X_tst = scaler.transform(X_trn), scaler.transform(X_tst)
        regressor.fit(X_trn, y_trn)
        y_pred = regressor.predict(X_tst)
        print(pearsonr(y_true, y_pred))
        preds.extend(y_pred)
        gt.extend(y_true)
    return gt, preds


# In[801]:

X = np.transpose([all_features[feat] for feat in acc_feats])
X = np.concatenate([X, np.asarray(ss_arr)], axis=1)


# In[802]:

from sklearn.ensemble import RandomForestRegressor
def get_regressor(): return RandomForestRegressor(n_estimators=100, random_state=101)
gt, preds = run_cv_test(X, get_regressor, normalize=1)
print(pearsonr(gt, preds))
len(gt)


# In[803]:

from sklearn.svm import SVR
def get_regressor(): return SVR(kernel='rbf')
gt, preds = run_cv_test(X, get_regressor, normalize=1)
print(pearsonr(gt, preds))
len(gt)


# In[804]:


def run_cv_test(X, alpha=0.2, normalize=1):
    gt, preds = [], []
    for group in [G1, G2, G3, G4, G5]:
        indx_tst = df.Protein.isin(group)
        indx_trn = np.logical_not(indx_tst)
        y_trn = df.DDG[indx_trn]
        y_true = df.DDG[indx_tst]
        X_trn = X[indx_trn]
        X_tst = X[indx_tst]
        rf = RandomForestRegressor(n_estimators=50, random_state=101)
        svr = SVR(kernel='rbf')
        if normalize == 1:
            scaler = StandardScaler()
            scaler.fit(X_trn)
            X_trn, X_tst = scaler.transform(X_trn), scaler.transform(X_tst)
        svr.fit(X_trn, y_trn)
        rf.fit(X_trn, y_trn)
        y_pred_svr = svr.predict(X_tst)
        y_pred_rf = rf.predict(X_tst)
        y_pred = alpha * y_pred_svr + (1-alpha) * y_pred_rf
        print(pearsonr(y_true, y_pred))
        preds.extend(y_pred)
        gt.extend(y_true)
    return gt, preds


# In[805]:


gt, preds = run_cv_test(X, normalize=1)
print(pearsonr(gt, preds))
len(gt)


# In[806]:


cp_b = np.asarray(all_features[('uniform', 'CP_B', 'BASU010101', 7)])
cp_a = np.asarray(all_features[('uniform', 'CP_A', 'BASU010101', 7)])
ei = np.asarray(all_features[('EI', 'SKOJ970101')])
ddg = skempi_df.DDG


# In[807]:


c1 = pearsonr(ei, ddg)[0]
c2 = pearsonr(cp_a, ddg)[0]
c3 = pearsonr(cp_b, ddg)[0]
s = c1 + c2 + c3
a1 = c1/s
a2 = c2/s
a3 = c3/s
print(c1, c2, c3)


# In[808]:

ddg_hat = np.multiply(a1, ei) + np.multiply(a2, cp_a) + np.multiply(a3, cp_b)
pearsonr(ddg_hat, ddg)
