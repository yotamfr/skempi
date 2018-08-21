import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim

import os
from os import path as osp

from skempi_utils import *
from pytorch_utils import *

from tempfile import gettempdir
from itertools import combinations as comb
from scipy.stats import pearsonr

import random
random.seed(0)

LR = 0.01
USE_CUDA = True
BATCH_SIZE = 32


class RegressorDDG(nn.Module):

    def __init__(self, input_size=10, output_size=1, dropout=0.1):
        super(RegressorDDG, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 100),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(100, 100),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(100, output_size)
        )

    def forward(self, x):
        return self.model(x)


def get_loss(ddg_hat, mddg_hat, ddg_gt):
    mse = nn.MSELoss().cuda()
    consistency = mse(-ddg_hat, mddg_hat)
    completeness = mse((ddg_hat-mddg_hat) * 0.5, ddg_gt.unsqueeze(1))
    return consistency + completeness


def get_loss2(ddg_hat, ddg_gt):
    mse = nn.MSELoss().cuda()
    loss = mse(ddg_hat, ddg_gt)
    return loss


def evaluate(model, batch_generator, length_xy):
    model.eval()
    pbar = tqdm(total=length_xy, desc="calculation...")
    err, i = 0, 0
    ddg_hat, ddg_gt = [], []
    for i, (x1, x2, y) in enumerate(batch_generator):
        o1 = model(x1)
        o2 = model(x2)
        y_hat = (o1-o2).view(-1)
        loss = get_loss2(y_hat, y)
        err += loss.item()
        ddg_hat.extend(y_hat.data.cpu().numpy())
        ddg_gt.extend(y.data.cpu().numpy())
        pbar.update(len(y))
    cor_pos, _ = pearsonr(np.asarray(ddg_hat), np.asarray(ddg_gt))
    cor_neg, _ = pearsonr(-np.asarray(ddg_hat), -np.asarray(ddg_gt))
    pbar.set_description("COR_POS:%.2f, COR_NEG:%.2f" % (cor_pos, cor_neg))
    pbar.close()
    return err / (i + 1)


def train(model, opt, adalr, batch_generator, length_xy):

    model.train()

    pbar = tqdm(total=length_xy, desc="calculating...")

    err = 0

    for i, (x1, x2, ddg) in enumerate(batch_generator):

        opt.zero_grad()
        o1 = model(x1)
        o2 = model(x2)
        loss = get_loss(o1, o2, ddg)
        adalr.update(loss.item())
        err += loss.item()
        loss.backward()
        opt.step()
        lr, e = adalr.lr, err/(i + 1)
        pbar.set_description("Training Loss:%.4f, LR: %.4f" % (e, lr))
        pbar.update(len(ddg))

    pbar.close()


class Loader(object):

    def __init__(self, X_pos, X_neg, y, shuffle=True):
        self._curr = 0
        self._data = list(zip(X_pos, X_neg, y))
        if shuffle:
            indx = range(len(self._data))
            self._data = [self._data[i] for i in indx]

    def reset(self):
        self._curr = 0

    def __iter__(self):
        return self

    def next(self):
        if self._curr < len(self._data):
            x1, x2, ddg = self._data[self._curr]
            self._curr += 1
            return x1, x2, ddg
        else:
            raise StopIteration

    def __next__(self):
        return self.next()

    def __str__(self):
        return "<Loader: %d>" % len(self._data)

    def __len__(self):
        return len(self._data)


def batch_generator(loader, batch_size=BATCH_SIZE):

    def prepare_batch(x1, x2, ddg):
        x1_var = Variable(torch.FloatTensor(x1))
        x2_var = Variable(torch.FloatTensor(x2))
        ddg_var = Variable(torch.FloatTensor(ddg))

        if USE_CUDA:
            x1_var = x1_var.cuda()
            x2_var = x2_var.cuda()
            ddg_var = ddg_var.cuda()

        return x1_var, x2_var, ddg_var

    stop = False
    while not stop:
        batch = []
        while len(batch) < batch_size:
            try:
                batch.append(next(loader))
            except StopIteration:
                stop = True
                break
        if len(batch) == 0:
            break
        x1, x2, ddg = zip(*batch)
        yield prepare_batch(x1, x2, ddg)


def add_arguments(parser):

    parser.add_argument('-r', '--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument("-e", "--eval_every", type=int, default=1,
                        help="How often to evaluate on the validation set.")
    parser.add_argument("--num_epochs", type=int, default=20,
                        help="How many epochs to train the model?")
    parser.add_argument("-o", "--out_dir", type=str, required=False,
                        default=gettempdir(), help="Specify the output directory.")
    parser.add_argument("-s", "--seed", type=int, default=9898,
                        help="Sets the seed for generating random number.")


def split_data_train_test(g1, g2, X_pos, X_neg, ddg, ix_pos, ix_neg):
    group = list((g1, g2))
    rest = list(set([1, 2, 3, 4, 5]) - set((g1, g2)))
    indx_trn_pos = np.isin(ix_pos, rest)
    indx_trn_neg = np.isin(ix_neg, rest)
    loader_trn = Loader(X_pos[indx_trn_pos, :], X_neg[indx_trn_neg, :], ddg[indx_trn_pos])
    indx_tst_pos = np.isin(ix_pos, group)
    indx_tst_neg = np.isin(ix_neg, group)
    loader_val = Loader(X_pos[indx_tst_pos, :], X_neg[indx_tst_neg, :], ddg[indx_tst_pos])
    return loader_trn, loader_val


def run_cv(data_pos, data_neg):
    X_pos, y, ix_pos, _, _ = [np.asarray(d) for d in zip(*data_pos)]
    X_neg, _, ix_neg, _, _ = [np.asarray(d) for d in zip(*data_neg)]

    preds_data, groups_data = [], []

    for i, pair in enumerate(comb(range(NUM_GROUPS), 2)):

        g1, g2 = np.asarray(pair) + 1
        loader_trn, loader_val = split_data_train_test(g1, g2, X_pos, X_neg, y, ix_pos, ix_neg)
        net = RegressorDDG()
        opt = optim.Adamax(net.parameters(), lr=LR)

        ckptpath = args.out_dir

        model_summary(net)

        init_epoch = 0
        num_epochs = args.num_epochs

        # optionally resume from a checkpoint
        if args.resume:
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '%s'" % args.resume)
                checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage)
                init_epoch = checkpoint['epoch']
                net.load_state_dict(checkpoint['net'])
                opt.load_state_dict(checkpoint['opt'])
            else:
                print("=> no checkpoint found at '%s'" % args.resume)

        adalr = AdaptiveLR(opt, LR)

        # Move models to GPU
        if USE_CUDA:
            net = net.cuda()
        if USE_CUDA and args.resume:
            optimizer_cuda(opt)

        for epoch in range(init_epoch, num_epochs):

            train(net, opt, adalr, batch_generator(loader_trn), len(loader_trn))

            if epoch < num_epochs - 1 and epoch % args.eval_every != 0:
                continue

            loss = evaluate(net, batch_generator(loader_val), len(loader_val))

            if VERBOSE:
                print("[Epoch %d/%d] (Validation Loss: %.4f" % (epoch + 1, num_epochs, loss))

            save_checkpoint({
                'lr': adalr.lr,
                'epoch': epoch,
                'net': net.state_dict(),
                'opt': opt.state_dict()
            }, loss, "skempi", ckptpath)

            loader_val.reset()
            loader_trn.reset()


def records_to_xy(skempi_records, load_negative=False):
    data = []
    for record in tqdm(skempi_records, desc="records processed"):
        assert record.struct is not None
        r = reversed(record) if load_negative else record
        data.append([r.features(True), r.ddg, r.group, r.modelname, r.mutations])
    return data


if __name__ == "__main__":
    import pickle
    import argparse
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()

    out_dir = args.out_dir
    try:
        with open('%s/data_pos.pkl' % out_dir, 'rb') as f:
            data_pos = pickle.load(f)
        with open('%s/data_neg.pkl' % out_dir, 'rb') as f:
            data_neg = pickle.load(f)
        with open('%s/data_pos.pkl' % out_dir, 'w+b') as f:
            pickle.dump(data_pos, f)
        with open('%s/data_neg.pkl' % out_dir, 'w+b') as f:
            pickle.dump(data_neg, f)
    except IOError:
        skempi_structs = load_skempi_structs("../data/pdbs", compute_dist_mat=False)
        skempi_records = load_skempi_records(skempi_structs)
        data_pos = records_to_xy(skempi_records, load_negative=False)
        data_neg = records_to_xy(skempi_records, load_negative=True)
        with open('%s/data_pos.pkl' % out_dir, 'w+b') as f:
            pickle.dump(data_pos, f)
        with open('%s/data_neg.pkl' % out_dir, 'w+b') as f:
            pickle.dump(data_neg, f)

    run_cv(data_pos, data_neg)
