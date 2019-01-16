import torch
from torch import nn
from torch import optim
from tqdm import tqdm

from torch_utils import *
from skempi_lib import *
from loader import *

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_bins = 301
DDG = np.asarray(s2648_df.DDG.values.tolist() + varib_df.DDG.values.tolist())
Min, Max = min(min(DDG), min(-DDG)), max(max(DDG), max(-DDG))
bins = np.linspace(Min - 0.1, Max + 0.1, num_bins)
bins_tensor = torch.tensor(bins, dtype=torch.float, device=device)

DEBUG = False
BATCH_SIZE = 32
LR = 0.001

np.random.seed(101)


class Conv1(nn.Module):
    def __init__(self, nc, ngf, ndf, latent_variable_size):
        super(Conv1, self).__init__()

        self.nc = nc
        self.ngf = ngf
        self.ndf = ndf
        self.latent_variable_size = latent_variable_size

        # encoder
        self.e1 = nn.Conv3d(nc, ndf, 5, 1, 1)
        self.bn1 = nn.BatchNorm3d(ndf)

        self.e2 = nn.Conv3d(ndf, ndf * 2, 3, 1, 1)
        self.bn2 = nn.BatchNorm3d(ndf * 2)

        self.e3 = nn.Conv3d(ndf * 2, ndf * 4, 3, 1, 1)
        self.bn3 = nn.BatchNorm3d(ndf * 4)

        self.e4 = nn.Conv3d(ndf * 4, ndf * 8, 3, 1, 1)
        self.bn4 = nn.BatchNorm3d(ndf * 8)

        self.fc1 = nn.Linear(ndf * 8, latent_variable_size)
        self.fc2 = nn.Linear(ndf * 8, latent_variable_size)

        # aa classification
        self.fc3 = nn.Linear(ndf * 8, 20)

        ## DDG classification
        self.fc4 = nn.Linear(ndf * 8, latent_variable_size)
        self.fc5 = nn.Linear(latent_variable_size + 20 * 3, num_bins)

        self.leakyrelu = nn.LeakyReLU(0.2, inplace=True)
        self.mp = nn.MaxPool3d(2)

    def encode(self, x):
        h1 = self.mp(self.leakyrelu(self.bn1(self.e1(x))))
        h2 = self.mp(self.leakyrelu(self.bn2(self.e2(h1))))
        h3 = self.mp(self.leakyrelu(self.bn3(self.e3(h2))))
        h4 = self.mp(self.leakyrelu(self.bn4(self.e4(h3))))
        return h4.view(h4.size(0), -1)

    def forward(self, x, w, m):
        enc = self.encode(x)
        aa = self.fc3(enc)
        h = self.fc4(enc)
        ddg = self.fc5(torch.cat([h, aa, w, m], 1))
        return aa, torch.sigmoid(ddg)


def pearsonr_torch(out, tgt):
    x = bins_tensor.gather(0, torch.argmax(tgt, 1))
    y = bins_tensor.gather(0, torch.argmax(out > 0.5, 1))
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    xm = x.sub(mean_x)
    ym = y.sub(mean_y)
    r_num = xm.dot(ym)
    r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
    r_val = r_num / r_den
    return r_val


def get_loss(y_hat1, y_hat2, y1, y2, criterion=nn.BCELoss()):
    loss1 = criterion(y_hat1, y1)
    loss2 = criterion(y_hat2, y2)
    return loss1 + loss2


def preprocess_batch(records, start, end):
    y1 = torch.tensor([bins > rec.ddg for rec in records[start:end]], dtype=torch.float, device=device)
    y2 = torch.tensor([bins > -rec.ddg for rec in records[start:end]], dtype=torch.float, device=device)
    vox1 = [load_single_position(rec.struct, rec.struct.get_residue(rec.mutations[0]), rad=16., reso=1.25)[1]
            for rec in records[start:end]]
    vox2 = [load_single_position(rec.mutant, rec.mutant.get_residue(rec.mutations[0]), rad=16., reso=1.25)[1]
            for rec in records[start:end]]
    w = torch.tensor([amino_acids.index(rec.mutations[0].w) for rec in records[start:end]], dtype=torch.long,
                     device=device)
    m = torch.tensor([amino_acids.index(rec.mutations[0].m) for rec in records[start:end]], dtype=torch.long,
                     device=device)
    w = torch.zeros([end - start, 20], dtype=torch.float, device=device).scatter_(1, w.unsqueeze(1), 1)
    m = torch.zeros([end - start, 20], dtype=torch.float, device=device).scatter_(1, m.unsqueeze(1), 1)
    x1 = torch.cat([v.unsqueeze(0) for v in vox1], 0)
    x2 = torch.cat([v.unsqueeze(0) for v in vox2], 0)
    return x1, x2, y1, y2, w, m


def train(records, model, opt, batch_size):
    model.train()
    pbar = tqdm(range(len(records)), desc="calculating...")
    n_iter, err, acc = 1., 0., 0.
    for ibatch in range(0, len(records), batch_size):
        start = ibatch
        end = min(batch_size + ibatch, len(records))
        x1, x2, y1, y2, w, m = preprocess_batch(records, start, end)
        opt.zero_grad()
        _, y_hat1 = model(x1, w, m)
        _, y_hat2 = model(x2, m, w)
        loss = get_loss(y_hat1, y_hat2, y1, y2)
        pcor = pearsonr_torch(y_hat1, y1)
        err += loss.item()
        acc += pcor.item()
        loss.backward()
        opt.step_and_update_lr(loss.item())
        lr, e, a = opt.lr, err / n_iter, acc / n_iter
        pbar.set_description("Training Loss:%.4f, Perasonr: %.4f, LR: %.5f" % (e, a, lr))
        pbar.update(end-start)
        n_iter += 1
    pbar.close()


def evaluate(records, model, batch_size):
    model.eval()
    pbar = tqdm(range(len(records)), desc="calculating...")
    n_iter, err, acc = 1., 0., 0.
    for ibatch in range(0, len(records), batch_size):
        start = ibatch
        end = min(batch_size + ibatch, len(records))
        x1, x2, y1, y2, w, m = preprocess_batch(records, start, end)

        _, y_hat1 = model(x1, w, m)
        _, y_hat2 = model(x2, m, w)
        loss = get_loss(y_hat1, y_hat2, y1, y2)
        pcor = pearsonr_torch(y_hat1, y1)
        err += loss.item()
        acc += pcor.item()
        lr, e, a = opt.lr, err / n_iter,  acc / n_iter
        pbar.set_description("Validation Loss:%.4f, Perasonr: %.4f, LR: %.5f" % (e, a, lr))
        pbar.update(end-start)
        n_iter += 1
    pbar.close()
    return err/n_iter


if __name__ == "__main__":

    net = Conv1(nc=24, ngf=64, ndf=64, latent_variable_size=256)
    checkpoint = torch.load('../models/beast_1.45872.tar', map_location=lambda storage, loc: storage)
    load_checkpoint(net, checkpoint)
    net.to(device)

    dfs_freeze(net.bn1)
    dfs_freeze(net.bn2)
    dfs_freeze(net.bn3)
    dfs_freeze(net.bn4)
    dfs_freeze(net.e1)
    dfs_freeze(net.e2)
    dfs_freeze(net.e3)
    dfs_freeze(net.e4)
    dfs_freeze(net.fc1)
    dfs_freeze(net.fc2)
    dfs_freeze(net.fc3)

    lim = 500 if DEBUG else None
    varib = load_protherm(varib_df[:lim], PROTHERM_PDBs, True, 0)
    s2648 = load_protherm(s2648_df[:lim], PROTHERM_PDBs, True, 0)
    data = varib + s2648

    r = int(len(data) * 0.8)
    data = np.random.permutation(data)
    training, validation = data[:r], data[r:]

    opt = ScheduledOptimizer(optim.Adam(net.parameters(), lr=LR), LR, num_iterations=100)

    num_epochs = 50
    for epoch in range(num_epochs):
        train(training, net, opt, batch_size=BATCH_SIZE)
        loss = evaluate(validation, net, batch_size=BATCH_SIZE)
        print("[Epoch %d/%d] Validation Loss: %.4f" % (epoch + 1, num_epochs, loss))
