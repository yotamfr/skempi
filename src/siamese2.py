import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from tempfile import gettempdir
from tqdm import tqdm

from pytorch_utils import *
from reader_utils import *
from siamese_utils import *


def batch_generator(dataset, batch_size=BATCH_SIZE):

    def prepare_batch(dat1, dat2, lbl):
        dat_var1 = torch.tensor(np.asarray(dat1), dtype=torch.float, device=device)
        dat_var2 = torch.tensor(np.asarray(dat2), dtype=torch.float, device=device)
        lbl_var1 = torch.tensor(np.digitize(lbl, bins), dtype=torch.long, device=device)
        lbl_var2 = torch.tensor(np.digitize(-lbl, bins), dtype=torch.long, device=device)
        return dat_var1, dat_var2, lbl_var1, lbl_var2

    stop = False
    while not stop:
        X, y = [], []
        while len(y) < batch_size:
            try:
                box1, box2, ddg = next(dataset)
                X.append([box1, box2])
                y.append(ddg)
            except StopIteration:
                stop = True
                break
        if len(y) == 0:
            break
        X1, X2 = zip(*X)
        # yield prepare_batch(X1, X2, (np.asarray(y)-mu)/sigma)
        yield prepare_batch(X1, X2, np.asarray(y))


class CNN3D(nn.Module):     # Generator Code

    def __init__(self):
        super(CNN3D, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 16 x 16 x 16
            nn.Conv3d(nc, nenc, kernel_size=4, stride=2),
            nn.LeakyReLU(0.2, inplace=True),
            # input is (nenc) x 8 x 8 x 8
            nn.Conv3d(nenc, nenc*2, kernel_size=4, stride=2),
            nn.LeakyReLU(0.2, inplace=True),
            # input is (nenc*2) x 4 x 4 x 4
            nn.Conv3d(nenc*2, nenc*4, kernel_size=2, stride=2),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Conv3d(nenc*4, nenc*8, kernel_size=2, stride=2),
            # nn.ReLU(inplace=True),
            Flatten(),
            nn.Linear(nenc*4, num_bins),
        )
        self.main.to(device)

    def forward(self, input):
        return self.main(input)


def pearsonr_torch(out, tgt):
    bins_tensor = torch.tensor(bins, dtype=torch.float, device=device)
    x = bins_tensor.gather(0, tgt)
    y = bins_tensor.gather(0, torch.argmax(out, 1))
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    xm = x.sub(mean_x)
    ym = y.sub(mean_y)
    r_num = xm.dot(ym)
    r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
    r_val = r_num / r_den
    return r_val


def get_loss(y_hat1, y_hat2, y1, y2, criterion=FocalLoss(gamma=2)):
    loss1 = criterion(y_hat1, y1)
    loss2 = criterion(y_hat2, y2)
    return loss1 + loss2


def train(model, opt, adalr, batch_generator, length_xy, epoch):

    model.train()

    pbar = tqdm(total=length_xy, desc="calculating...")

    err, acc = 0., 0.

    for i, (x1, x2, y1, y2) in enumerate(batch_generator):

        x_drct = torch.cat([x1, x2], 1)
        x_rvrs = torch.cat([x2, x1], 1)
        opt.zero_grad()
        o = model(torch.cat([x_drct, x_rvrs], 0))
        o1 = o[:len(x_drct), :]
        o2 = o[len(x_drct):, :]
        loss = get_loss(o1, o2, y1, y2)
        pcor = pearsonr_torch(o1, y1)

        n_iter = epoch*length_xy + i
        writer.add_scalars('Siamese2/Loss', {"train": loss.item()}, n_iter)
        writer.add_scalars('Siamese2/PCC', {"train": pcor.item()}, n_iter)

        adalr.update(loss.item())
        err += loss.item()
        acc += pcor.item()
        loss.backward()
        opt.step()
        lr, e, a = adalr.lr, err/(i + 1), acc/(i + 1)
        pbar.set_description("Training Loss:%.4f, Perasonr: %.4f, LR: %.4f" % (e, a, lr))
        pbar.update(len(y1))

    pbar.close()


def evaluate(model, batch_generator, length_xy, n_iter):
    model.eval()
    pbar = tqdm(total=length_xy, desc="calculation...")
    err, acc = 0., 0.
    for i, (x1, x2, y1, y2) in enumerate(batch_generator):

        x_drct = torch.cat([x1, x2], 1)
        x_rvrs = torch.cat([x2, x1], 1)
        o = model(torch.cat([x_drct, x_rvrs], 0))
        o1 = o[:len(x_drct), :]
        o2 = o[len(x_drct):, :]
        loss = get_loss(o1, o2, y1, y2)
        pcor = pearsonr_torch(o1, y1)
        err += loss.item()
        acc += pcor.item()
        e, a = err/(i + 1), acc/(i + 1)
        pbar.set_description("Validation Loss:%.4f, Perasonr: %.4f" % (e, a))
        pbar.update(len(y1))

    writer.add_scalars('Siamese2/Loss', {"valid": e}, n_iter)
    writer.add_scalars('Siamese2/PCC', {"valid": a}, n_iter)
    pbar.close()
    return e, pcor


def add_arguments(parser):
    parser.add_argument("--mongo_url", type=str, default='mongodb://rack-jonathan-g04:27017', help="Supply the URL of MongoDB")


if __name__ == "__main__":
    import argparse
    from pymongo import MongoClient

    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    client = MongoClient(args.mongo_url)

    records_trn, records_tst = load_records(ratio_train_test=0.8)
    cache_trn = Cache(client['skempi']['siamese2_train'], serialize=serialize, deserialize=deserialize)
    cache_val = Cache(client['skempi']['siamese2_valid'], serialize=serialize, deserialize=deserialize)
    dataset_trn = SiameseDataset(records_trn, cache_trn)
    dataset_tst = SiameseDataset(records_tst, cache_val)

    net = CNN3D()
    opt = optim.SGD(net.parameters(), lr=LR, momentum=0.9, nesterov=True)
    adalr = AdaptiveLR(opt, LR, num_iterations=200)

    num_epochs = 20
    init_epoch = 0
    for epoch in range(init_epoch, num_epochs):

        train(net, opt, adalr, batch_generator(dataset_trn), len(dataset_trn), epoch)

        loss, _ = evaluate(net, batch_generator(dataset_tst), len(dataset_tst), (epoch+1)*len(dataset_trn))

        print("[Epoch %d/%d] (Validation Loss: %.4f" % (epoch + 1, num_epochs, loss))

        dataset_trn.reset()
        dataset_tst.reset()
