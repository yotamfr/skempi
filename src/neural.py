import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from skempi_lib import *
from torch_utils import *
from tensorboardX import SummaryWriter

writer = SummaryWriter('runs')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(1)
DEBUG = False


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

    def fit(self, X, y, valid=None, prefix="anonym", batch_size=96, epochs=1000, l_rate=0.01):

        torch.manual_seed(1)
        np.random.seed(1)

        opt = ScheduledOptimizer(optim.Adam(self.parameters(), lr=l_rate), l_rate, num_iterations=10)

        n_iter = 0
        for epoch in range(epochs):
            epoch += 1
            i = 0
            X, y = shuffle(X, y)
            while i < len(X):
                start = i
                end = min(i + batch_size, len(X))
                batch_X = X[start:end]
                batch_y = y[start:end]
                self.train()
                opt.zero_grad()
                loss, cor = self.get_loss(batch_X, batch_y)
                # writer.add_histogram('%s/Hist' % (prefix,), self.predict(X), epoch)
                writer.add_scalars('%s/Loss' % (prefix,), {"train": loss.item()}, n_iter)
                writer.add_scalars('%s/PCC' % (prefix,), {"train": cor.item()}, n_iter)
                # writer.add_scalars('%s/LR' % (prefix,), {"train": adalr.lr}, n_iter)
                # writer.add_embedding(self.emb_aa.weight.cpu().data.numpy(), global_step=n_iter,
                #                      metadata=['X']+amino_acids, tag="amino_acids")
                loss.backward()  # back props
                opt.step_and_update_lr(loss.item())  # update the parameters
                i += batch_size
                n_iter += 1

            if valid is not None:
                self.eval()
                X_val, y_val = valid
                loss, cor = self.get_loss(X_val, y_val)
                if DEBUG and (not (epoch + 1) % 1):
                    print('[%d/%d] LR %.6f, Loss %.3f, PCC %.3f' %
                          (epoch + 1, epochs, opt.lr, loss.item(), cor.item()))
                writer.add_scalars('%s/Loss' % (prefix,), {"valid": loss.item()}, n_iter)
                writer.add_scalars('%s/PCC' % (prefix,), {"valid": cor.item()}, n_iter)

            if opt.lr == opt.min_lr:
                print("stopped after %d epochs" % epoch)
                break

    def predict(self, X):
        self.eval()
        tensor_x = torch.tensor(X, dtype=torch.float, device=device)
        return self.forward(tensor_x).view(-1).cpu().data.numpy()


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


class ConsistentLinearModel(Model):
    def __init__(self, input_dim, output_dim=1):
        super(ConsistentLinearModel, self).__init__()
        self.r = nn.Sequential(
            nn.Linear(input_dim, output_dim),
        )
        self.r.to(device)

    def forward(self, x):
        return self.r(x).view(-1)

    def get_loss(self, X, y):
        inp = torch.tensor(X, dtype=torch.float, device=device)
        lbl = torch.tensor(y, dtype=torch.float, device=device)
        y_hat_p = self.forward(inp).view(-1)
        y_hat_m = self.forward(-inp).view(-1)

        mse = nn.MSELoss()
        completeness0 = mse(0.5 * (y_hat_p - y_hat_m), lbl)
        consistency0 = mse(-y_hat_p, y_hat_m)

        loss = completeness0 + consistency0
        return loss, pearsonr_torch(y_hat_p, lbl)


class BiasConsistentLinearModel(Model):
    def __init__(self, input_dim, output_dim=1):
        super(BiasConsistentLinearModel, self).__init__()
        self.r = nn.Sequential(
            nn.Linear(input_dim, output_dim),
        )
        self.r.to(device)

    def forward(self, x):
        return self.r(x).view(-1)

    def get_loss(self, X, y):
        inp = torch.tensor(X, dtype=torch.float, device=device)
        lbl = torch.tensor(y, dtype=torch.float, device=device)
        y_hat_p = self.forward(inp).view(-1)
        y_hat_m = self.forward(-inp).view(-1)

        mse = nn.MSELoss()
        completeness = mse(0.5 * (y_hat_p - y_hat_m), lbl)
        consistency = mse(-y_hat_p, y_hat_m)

        b_size = y_hat_m.size(0)
        b_consistency = (y_hat_p - torch.mean(y_hat_m)).sum().div(2*b_size).abs()

        loss = completeness + consistency + b_consistency
        return loss, pearsonr_torch(y_hat_p, lbl)


class BiasConsistentMultiLayerModel(Model):
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super(BiasConsistentMultiLayerModel, self).__init__()
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

        b_size = y_hat_m.size(0)
        b_consistency = (y_hat_p - torch.mean(y_hat_m)).sum().div(2*b_size).pow(2)

        loss = completeness0 + consistency0 + completeness2 + consistency2 + b_consistency
        return loss, pearsonr_torch(y_hat_p, lbl)


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


class CompoundModel(Model):

    def __init__(self):
        super(CompoundModel, self).__init__()

    def forward(self, x):
        x = x.cpu().data.numpy()
        nanix = np.isnan(x).any(axis=1)
        if (~nanix).sum() == 0:
            return self.m2(torch.tensor(x[:, 1:], dtype=torch.float, device=device))
        if nanix.sum() == 0:
            return self.m1(torch.tensor(x[:, :], dtype=torch.float, device=device))
        o = np.zeros(len(x))
        o1 = self.m1(torch.tensor(x[~nanix, :], dtype=torch.float, device=device))
        o[~nanix] = o1.cpu().data.numpy()
        o2 = self.m2(torch.tensor(x[nanix, 1:], dtype=torch.float, device=device))
        o[nanix] = o2.cpu().data.numpy()
        return torch.tensor(o, dtype=torch.float, device=device)

    def get_loss(self, x, y):
        nanix = np.isnan(x).any(axis=1)
        l1, pcor1 = self.m1.get_loss(x[~nanix, :], y[~nanix])
        if nanix.sum() == 0: return l1, pcor1
        l2, pcor2 = self.m2.get_loss(x[:, 1:], y[:])
        return l1 + l2, pcor1 + pcor2


class CompoundMultiLayerModel(CompoundModel):

    def __init__(self, input_dim, hidden_dim):
        super(CompoundMultiLayerModel, self).__init__()
        self.m1 = MultiLayerModel(input_dim, hidden_dim)
        self.m2 = MultiLayerModel(input_dim - 1, hidden_dim)
        self.m1.to(device)
        self.m2.to(device)


class CompoundLinearModel(CompoundModel):

    def __init__(self, input_dim):
        super(CompoundLinearModel, self).__init__()
        self.m1 = LinearRegressionModel(input_dim, 1)
        self.m2 = LinearRegressionModel(input_dim - 1, 1)
        self.m1.to(device)
        self.m2.to(device)


class LinearModelSklearn(object):

    def __init__(self, init_cmd="linear_model.Ridge(alpha=1.0)", import_cmd="from sklearn import linear_model"):
        exec import_cmd
        self.m = eval(init_cmd)

    def fit(self, x, y, valid=None, prefix="anonym"):
        self.m.fit(x, y)

    def predict(self, x):
        o = self.m.predict(x)
        return o


class CompoundModelSklearn(object):

    def __init__(self, init_cmd="linear_model.Ridge(alpha=1.0)", import_cmd="from sklearn import linear_model"):
        exec import_cmd
        self.m1 = eval(init_cmd)
        self.m2 = eval(init_cmd)

    def fit(self, x, y, valid=None, prefix="anonym"):
        nanix = np.isnan(x).any(axis=1)
        self.m1.fit(x[~nanix, :], y[~nanix])
        self.m2.fit(x[:, 1:], y[:])

    def predict(self, x):
        nanix = np.isnan(x).any(axis=1)
        if (~nanix).sum() == 0:
            return self.m2.predict(x[:, 1:])
        if nanix.sum() == 0:
            return self.m1.predict(x[:, :])
        o = np.zeros(len(x))
        o[~nanix] = self.m1.predict(x[~nanix, :])
        o[nanix] = self.m2.predict(x[nanix, 1:])
        return o


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


class LinearRegressionModel2(Model):
    def __init__(self):
        super(LinearRegressionModel2, self).__init__()
        self.model = nn.Linear(5, 6)
        self.model.to(device)
        self.emb_aa = nn.Embedding(21, 3)
        self.emb_aa.to(device)

    def forward(self, inputs, indices):
        feats, prof, pos, mut = inputs['Feats'], inputs['Prof'], inputs['Pos'], inputs['Mut']
        wt = self.emb_aa(torch.tensor(np.asarray(pos), dtype=torch.long, device=device))
        mt = -self.emb_aa(torch.tensor(np.asarray(mut), dtype=torch.long, device=device))
        ff = torch.tensor(np.asarray(feats), dtype=torch.float, device=device)
        ddg = self.model(ff).unsqueeze(1).bmm(torch.cat([wt, mt], 2).transpose(1, 2)).squeeze(1)
        out = torch.zeros((len(indices), 1), device=device)
        for i, ixs in enumerate(indices):
            for j in ixs: out[i, :] += ddg[j, :]
        return out

    def get_loss(self, X, y):
        mse = nn.MSELoss()
        inp1, indices = flatten(X)
        inp2 = {'Feats': [-s for s in inp1['Feats']], 'Prof': inp1['Prof'], 'Mut': inp1['Pos'], 'Pos': inp1['Mut']}
        lbls = torch.tensor(y, dtype=torch.float, device=device)
        y_hat_p = self.forward(inp1, indices).view(-1)
        y_hat_m = self.forward(inp2, indices).view(-1)
        completeness0 = mse(0.5 * (y_hat_p - y_hat_m), lbls)
        consistency0 = mse(-y_hat_p, y_hat_m)
        loss = completeness0 + consistency0
        return loss, pearsonr_torch(y_hat_p, lbls)

    def predict(self, X):
        self.eval()
        inputs, indices = flatten(X)
        return self.forward(inputs, indices).view(-1).cpu().data.numpy()


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
        loss = mse(y_hat_p, lbl)
        return loss, pearsonr_torch(y_hat_p, lbl)

    def predict(self, X):
        self.eval()
        inputs1, indices = flatten(X)
        return self.forward(inputs1, indices).view(-1).cpu().data.numpy()

    def preprocess(self, inputs):
        selection_len = self.num_intact
        interactions = inputs['IntAct']
        env = [[ii[j].descriptor if j < len(ii) else Interaction.pad()
                for j in range(selection_len)] for ii in interactions]
        mt, aa1, aa2, at1, at2, ap1, ap2, dd, p1, p2, pm, ac11, ac12, ac21, ac22 = zip(
            *[zip(*e[:selection_len]) for e in env])
        mt = self.emb_aa(torch.tensor(np.asarray(mt), dtype=torch.long, device=device))
        aa1 = self.emb_aa(torch.tensor(np.asarray(aa1), dtype=torch.long, device=device))
        aa2 = self.emb_aa(torch.tensor(np.asarray(aa2), dtype=torch.long, device=device))
        at1 = self.emb_atm(torch.tensor(np.asarray(at1), dtype=torch.long, device=device))
        at2 = self.emb_atm(torch.tensor(np.asarray(at2), dtype=torch.long, device=device))
        ap1 = self.emb_pos(torch.tensor(np.asarray(ap1), dtype=torch.long, device=device))
        ap2 = self.emb_pos(torch.tensor(np.asarray(ap2), dtype=torch.long, device=device))
        dd = torch.tensor(np.asarray(dd), dtype=torch.float, device=device).unsqueeze(2)
        p1 = torch.tensor(np.asarray(p1), dtype=torch.float, device=device).unsqueeze(2)
        p2 = torch.tensor(np.asarray(p2), dtype=torch.float, device=device).unsqueeze(2)
        pm = torch.tensor(np.asarray(pm), dtype=torch.float, device=device).unsqueeze(2)
        ac11 = torch.tensor(np.asarray(ac11), dtype=torch.float, device=device).unsqueeze(2)
        ac12 = torch.tensor(np.asarray(ac12), dtype=torch.float, device=device).unsqueeze(2)
        ac21 = torch.tensor(np.asarray(ac21), dtype=torch.float, device=device).unsqueeze(2)
        ac22 = torch.tensor(np.asarray(ac22), dtype=torch.float, device=device).unsqueeze(2)
        interactions = torch.cat([mt, aa1, aa2, at1, at2, ap1, ap2, dd, p1, p2, pm, ac11, ac12, ac21, ac22], 2)
        return interactions


class PhysicalModel1(PhysicalModel):
    def __init__(self):
        super(PhysicalModel1, self).__init__()
        self.num_intact = 10
        self.emb_atm = nn.Embedding(8, 2)
        self.emb_pos = nn.Embedding(60, 3)
        self.emb_aa = nn.Embedding(21, 3)
        self.emb_atm.to(device)
        self.emb_pos.to(device)
        self.emb_aa.to(device)

        self.weight = nn.Sequential(
            nn.Linear(27, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
        )

        #         self.attn = ScaledDotProductAttention(np.power(54, 0.5))
        self.embed = nn.Linear(27, 54)
        self.ddg = nn.Linear(54, 1)

    def forward(self, inputs, indices):
        interactions = self.preprocess(inputs)
        embed_interactions = self.embed(interactions)
        weights = F.softmax(self.weight(interactions), 1)
        summary_interactions = weights.transpose(1, 2).bmm(embed_interactions).squeeze(1)
        #         weighted_interactions, _ = self.attn(embed_interactions, embed_interactions, embed_interactions)
        #         summary_interactions = weighted_interactions.mean(1)
        ddg = self.ddg(summary_interactions)
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

        self.weight = nn.Sequential(
            nn.Linear(27, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
        )

        self.predictor1 = nn.Sequential(
            nn.Linear(27, 1),
        )

    def forward(self, inputs, indices):
        interactions = self.preprocess(inputs)
        ddg = self.predictor1(interactions)
        weights = F.softmax(self.weight(interactions), 1)
        ddg = weights.transpose(1, 2).bmm(ddg).squeeze(1)
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
    m1 = PhysicalModel1()
    loss, pcc = m1.get_loss(X, y)
    print(loss, pcc)
    X = np.asarray([get_neural_features(r.struct, r.mutations) for r in records])
    m2 = LinearRegressionModel2()
    loss, pcc = m2.get_loss(X, y)
    print(loss, pcc)
