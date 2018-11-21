import os
from os import path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from skempi_lib import *
from pytorch_utils import *
from tensorboardX import SummaryWriter

writer = SummaryWriter('runs')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'

DEBUG = True


def pearsonr_torch(x, y):
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    xm = x.sub(mean_x)
    ym = y.sub(mean_y)
    r_num = xm.dot(ym)
    r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
    r_val = r_num / r_den
    return r_val


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        raise NotImplementedError

    def get_loss(self, X, y):
        raise NotImplementedError

    def fit(self, X, y, valid=None, prefix="anonym", batch_size=1000, epochs=50):
        l_rate = 0.01
        optimiser = torch.optim.Adam(self.parameters(), lr=l_rate)
        adalr = AdaptiveLR(optimiser, l_rate, num_iterations=10)

        n_iter = 0
        for epoch in range(epochs):
            epoch += 1
            i = 0
            while i < len(X):
                start = i
                end = min(i + batch_size, len(X))
                batch_X = X[start:end]
                batch_y = y[start:end]
                self.train()
                optimiser.zero_grad()
                loss, cor = self.get_loss(batch_X, batch_y)
                # writer.add_histogram('%s/Hist' % (prefix,), self.predict(X), epoch)
                writer.add_scalars('%s/Loss' % (prefix,), {"train": loss.item()}, n_iter)
                writer.add_scalars('%s/PCC' % (prefix,), {"train": cor.item()}, n_iter)
                # writer.add_scalars('%s/LR' % (prefix,), {"train": adalr.lr}, n_iter)

                loss.backward()  # back props
                optimiser.step()  # update the parameters
                adalr.update(loss.item())
                i += batch_size
                n_iter += 1

            if valid is not None:
                self.eval()
                X_val, y_val = valid
                loss, cor = self.get_loss(X_val, y_val)
                if DEBUG and (not (epoch + 1) % 1):
                    print('[%d/%d] LR %.6f, Loss %.3f, PCC %.3f' %
                          (epoch + 1, epochs, adalr.lr, loss.item(), cor.item()))
                writer.add_scalars('%s/Loss' % (prefix,), {"valid": loss.item()}, n_iter)
                writer.add_scalars('%s/PCC' % (prefix,), {"valid": cor.item()}, n_iter)

            if adalr.lr == adalr.min_lr:
                break

    def predict(self, X):
        self.eval()
        return self.forward(torch.tensor(X, dtype=torch.float, device=device)).view(-1).cpu().data.numpy()


class LinearRegressionModel(Model):
    def __init__(self, input_dim, output_dim=1):
        super(LinearRegressionModel, self).__init__()
        self.model = nn.Linear(input_dim, output_dim)
        self.model.to(device)

    def forward(self, x):
        out = self.model(x).view(-1)
        return out

    def get_loss(self, X, y):
        criterion = nn.MSELoss()
        inputs = torch.tensor(X, dtype=torch.float, device=device)
        labels = torch.tensor(y, dtype=torch.float, device=device)
        outputs = self.forward(inputs).view(-1)
        loss = criterion(outputs, labels)
        return loss, pearsonr_torch(outputs, labels)


class MultiLayerModel(Model):
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super(MultiLayerModel, self).__init__()
        self.r1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh(),
        )
        self.r2 = nn.Sequential(
            nn.Linear(input_dim + 1, output_dim),
        )
        self.r1.to(device)
        self.r2.to(device)

    def forward(self, x):
        o = self.r2(torch.cat([x, self.r1(x)], 1))
        return o.view(-1)

    def get_loss(self, X, y):
        inp = torch.tensor(X, dtype=torch.float, device=device)
        lbl = torch.tensor(y, dtype=torch.float, device=device)
        y_hat_p = self.forward(inp).view(-1)
        y_hat_m = self.forward(-inp).view(-1)
        z_hat_p = self.r1(inp).view(-1)
        z_hat_m = self.r1(-inp).view(-1)

        mse = nn.MSELoss()
        completeness0 = mse(0.5 * (y_hat_p - y_hat_m), lbl)
        consistency0 = mse(-y_hat_p, y_hat_m)
        completeness2 = mse(0.5 * (z_hat_p - z_hat_m), torch.sign(lbl))
        consistency2 = mse(-z_hat_p, z_hat_m)

        loss = completeness0 + consistency0 + completeness2 + consistency2
        return loss, pearsonr_torch(y_hat_p, lbl)


def flatten(objects):
    indices = []
    o = {k: [] for k in objects[0]}
    for i, obj in enumerate(objects):
        l = len(obj.values()[0])
        for k in obj:
            assert len(obj[k]) == l
            o[k].extend(obj[k])
        indices.append(range(len(o[k]) - l, len(o[k])))
    return o, indices


def get_neural_features(struct, mutations):
    feats = dict()
    feats["IntAct"] = [get_interactions(struct, mut) for mut in mutations]
    feats["Pos"] = [[amino_acids.index(mut.w) + 1] for mut in mutations]
    feats["Mut"] = [[amino_acids.index(mut.m) + 1] for mut in mutations]
    feats["Acc"] = [ac_ratio(struct, mut.chain_id, mut.i) for mut in mutations]
    feats["Prof"] = [[struct.get_profile(mut.chain_id)[(mut.i, aa)] for aa in amino_acids] for mut in mutations]
    return feats


class PhysicalModel(Model):
    def __init__(self):
        super(PhysicalModel, self).__init__()

    def forward(self, inputs, indices):
        raise NotImplementedError

    def get_loss(self, X, y):
        mse = nn.MSELoss()
        inputs1, indices = flatten(X)
        lbl = torch.tensor(y, dtype=torch.float, device=device)
        y_hat_p = self.forward(inputs1, indices).view(-1)
        # y_hat_m = self.forward(inputs2, indices).view(-1)
        # completeness0 = mse(0.5 * (y_hat_p - y_hat_m), lbl)
        # consistency0 = mse(-y_hat_p, y_hat_m)
        # loss = completeness0 + consistency0
        # return loss, pearsonr_torch(y_hat_p, lbl)
        return 1-pearsonr_torch(y_hat_p, lbl), pearsonr_torch(y_hat_p, lbl)

    def predict(self, X):
        self.eval()
        inputs1, indices = flatten(X)
        return self.forward(inputs1, indices).view(-1).cpu().data.numpy()


class PhysicalModel1(PhysicalModel):
    def __init__(self):
        super(PhysicalModel, self).__init__()
        self.num_intact = 10
        self.emb_atm = nn.Embedding(8, 2)
        self.emb_pos = nn.Embedding(60, 3)
        self.emb_aa = nn.Embedding(21, 3)
        self.emb_atm.to(device)
        self.emb_pos.to(device)
        self.emb_aa.to(device)

        self.weight = nn.Sequential(
            nn.Linear(20, 3),
            nn.ReLU(),
            nn.Linear(3, 1),
        )
        self.weight.to(device)
        self.ddg = nn.Sequential(
            nn.Linear(20, 3),
            nn.Tanh(),
            nn.Linear(3, 1),
        )
        self.ddg.to(device)

    def forward(self, inputs, indices):
        interactions, pos, mut, acc = inputs['IntAct'], inputs['Pos'], inputs['Mut'], inputs['Acc']
        selection_len = self.num_intact
        env = [[i.descriptor for i in ii] + ([[0, 0, 0, 0, 0, 0, 0.0]] * (max(0, selection_len - len(ii)))) for ii in interactions]
        aa1, aa2, at1, at2, ap1, ap2, dd = zip(*[zip(*e[:selection_len]) for e in env])
        mt = [tuple(m * selection_len) for m in mut]
        mt = self.emb_aa(torch.tensor(np.asarray(mt), dtype=torch.long, device=device))
        cc = self.emb_aa(torch.tensor(np.asarray(aa1), dtype=torch.long, device=device))
        rr = self.emb_aa(torch.tensor(np.asarray(aa2), dtype=torch.long, device=device))
        trr = self.emb_atm(torch.tensor(np.asarray(at2), dtype=torch.long, device=device))
        prr = self.emb_pos(torch.tensor(np.asarray(ap2), dtype=torch.long, device=device))
        twt = self.emb_atm(torch.tensor(np.asarray(at1), dtype=torch.long, device=device))
        pwt = self.emb_pos(torch.tensor(np.asarray(ap1), dtype=torch.long, device=device))
        dd = torch.tensor(np.asarray(dd), dtype=torch.float, device=device).unsqueeze(2)
        interactions = torch.cat([mt, cc, pwt, twt, dd, rr, prr, trr], 2)
        weights = F.softmax(self.weight(interactions), 1)
        weighted_interactions = weights.transpose(1, 2).bmm(interactions).squeeze(1)
        # ac = torch.tensor(np.asarray(acc), dtype=torch.float, device=device)
        # inp = torch.cat([weighted_interactions, ac.unsqueeze(1)], 1)
        # ddg = self.ddg(inp)
        ddg = self.ddg(weighted_interactions)
        out = torch.zeros((len(indices), 1), device=device)
        for i, ixs in enumerate(indices):
            for j in ixs: out[i, :] += ddg[j, :]
        return out


class PhysicalModel2(PhysicalModel):
    def __init__(self):
        super(PhysicalModel2, self).__init__()
        self.num_intact = 10
        self.emb_atm = nn.Embedding(8, 2)
        self.emb_pos = nn.Embedding(60, 3)
        self.emb_aa = nn.Embedding(21, 3)
        self.emb_atm.to(device)
        self.emb_pos.to(device)
        self.emb_aa.to(device)

        self.predictor = nn.Sequential(
            nn.Linear(20, 5),
            nn.ReLU(),
            nn.Linear(5, 1),
        )
        self.predictor.to(device)

    def forward(self, inputs, indices):
        interactions, pos, mut, acc = inputs['IntAct'], inputs['Pos'], inputs['Mut'], inputs['Acc']
        selection_len = self.num_intact
        env = [[i.descriptor for i in ii] + ([[0, 0, 0, 0, 0, 0, 0.0]] * (max(0, selection_len - len(ii)))) for ii in interactions]
        aa1, aa2, at1, at2, ap1, ap2, dd = zip(*[zip(*e[:selection_len]) for e in env])
        mt = [tuple(m * selection_len) for m in mut]
        mt = self.emb_aa(torch.tensor(np.asarray(mt), dtype=torch.long, device=device))
        cc = self.emb_aa(torch.tensor(np.asarray(aa1), dtype=torch.long, device=device))
        rr = self.emb_aa(torch.tensor(np.asarray(aa2), dtype=torch.long, device=device))
        trr = self.emb_atm(torch.tensor(np.asarray(at2), dtype=torch.long, device=device))
        prr = self.emb_pos(torch.tensor(np.asarray(ap2), dtype=torch.long, device=device))
        twt = self.emb_atm(torch.tensor(np.asarray(at1), dtype=torch.long, device=device))
        pwt = self.emb_pos(torch.tensor(np.asarray(ap1), dtype=torch.long, device=device))
        dd = torch.tensor(np.asarray(dd), dtype=torch.float, device=device).unsqueeze(2)
        interactions = torch.cat([mt, cc, pwt, twt, dd, rr, prr, trr], 2)
        ddg = torch.sum(self.predcitor(interactions), 1)
        # ac = torch.tensor(np.asarray(acc), dtype=torch.float, device=device)
        out = torch.zeros((len(indices), 1), device=device)
        for i, ixs in enumerate(indices):
            for j in ixs: out[i, :] += ddg[j, :]
        return out


class PhysicalModel3(PhysicalModel):
    def __init__(self):
        super(PhysicalModel3, self).__init__()
        self.num_intact = 10
        self.emb_atm = nn.Embedding(8, 2)
        self.emb_pos = nn.Embedding(60, 3)
        self.emb_aa = nn.Embedding(21, 3)
        self.emb_atm.to(device)
        self.emb_pos.to(device)
        self.emb_aa.to(device)

        self.ddg = nn.Sequential(
            nn.Linear(self.num_intact*(13+1) + 6, self.num_intact),
            nn.Tanh(),
            nn.Linear(self.num_intact, 1),
        )
        self.ddg.to(device)

    def forward(self, inputs, indices):
        interactions, pos, mut, acc = inputs['IntAct'], inputs['Pos'], inputs['Mut'], inputs['Acc']
        selection_len = self.num_intact
        env = [[i.descriptor[1:] for i in ii] + ([[0, 0, 0, 0, 0, 0.0]] * (max(0, selection_len - len(ii)))) for ii in interactions]
        rr, twt, trr, pwt, prr, dd = zip(*[zip(*e[:selection_len]) for e in env])
        rr = self.emb_aa(torch.tensor(np.asarray(rr), dtype=torch.long, device=device))
        trr = self.emb_atm(torch.tensor(np.asarray(trr), dtype=torch.long, device=device))
        prr = self.emb_pos(torch.tensor(np.asarray(prr), dtype=torch.long, device=device))
        twt = self.emb_atm(torch.tensor(np.asarray(twt), dtype=torch.long, device=device))
        pwt = self.emb_pos(torch.tensor(np.asarray(pwt), dtype=torch.long, device=device))
        dd = torch.tensor(np.asarray(dd), dtype=torch.float, device=device).unsqueeze(2)
        interactions = torch.cat([twt, pwt, rr, prr, trr, dd], 2).view(-1, selection_len*(13+1))
        ac = torch.tensor(np.asarray(acc), dtype=torch.float, device=device)
        wt = self.emb_aa(torch.tensor(np.asarray(pos), dtype=torch.long, device=device))
        mt = self.emb_aa(torch.tensor(np.asarray(mut), dtype=torch.long, device=device))
        # inp = torch.cat([interactions, wt.squeeze(1), mt.squeeze(1), ac.unsqueeze(1)], 1)
        inp = torch.cat([interactions, wt.squeeze(1), mt.squeeze(1)], 1)
        ddg = self.ddg(inp)
        out = torch.zeros((len(indices), 1), device=device)
        for i, ixs in enumerate(indices):
            for j in ixs: out[i, :] += ddg[j, :]
        return out


if __name__ == "__main__":
    lim = 100
    df = skempi_df_v2
    records = load_skempi(df[df.version == 2].reset_index(drop=True)[:lim], SKMEPI2_PDBs, False)
    y = np.asarray([r.ddg for r in records])
    X = np.asarray([get_neural_features(r.struct, r.mutations) for r in records])
    m1 = PhysicalModel2()
    loss, pcc = m1.get_loss(X, y)
    print(loss, pcc)
