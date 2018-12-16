import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from tensorboardX import SummaryWriter

writer = SummaryWriter('runs')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from tempfile import gettempdir
from tqdm import tqdm

from vae import *
from loader import *
from skempi_lib import *
from pytorch_utils import *

BATCH_SIZE = 32
LR = 1e-3


def get_loss(x_hat, x, mu, logvar):
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    L1 = nn.L1Loss()(x_hat, x)
    return L1 + KLD


def train(model, opt, adalr, batch_generator, length_xy, epoch):

    model.train()

    pbar = tqdm(total=length_xy, desc="calculating...")

    err = 0.

    for i, (x, y) in enumerate(batch_generator):
        opt.zero_grad()
        y_hat, mu, logvar = model(x)
        loss = get_loss(y_hat, y, mu, logvar)
        n_iter = epoch*length_xy + i
        writer.add_scalars('VAE/Loss', {"train": loss.item()}, n_iter)
        adalr.update(loss.item())
        err += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        loss.backward()
        opt.step()
        lr, e = adalr.lr, err/(i + 1)
        pbar.set_description("Training Loss:%.5f, LR: %.5f" % (e, lr))
        pbar.update(len(y))

    pbar.close()


def evaluate(model, batch_generator, length_xy, n_iter):
    model.eval()
    pbar = tqdm(total=length_xy, desc="calculation...")
    err = 0.
    for i, (x, y) in enumerate(batch_generator):
        y_hat, mu, logvar = model(x)
        loss = get_loss(y_hat, y, mu, logvar)
        err += loss.item()
        e = err/(i + 1)
        pbar.set_description("Validation Loss:%.5f" % (e,))
        pbar.update(len(y))
    writer.add_scalars('VAE/Loss', {"valid": e}, n_iter)
    pbar.close()
    return e


def add_arguments(parser):
    parser.add_argument("--mongo_url", type=str, default='mongodb://rack-jonathan-g04:27017', help="Supply the URL of MongoDB")


if __name__ == "__main__":
    import argparse
    from pymongo import MongoClient

    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    client = MongoClient(args.mongo_url)

    net = VAE(nc=25, ngf=64, ndf=64, latent_variable_size=100)
    net.to(device)
    # opt = optim.SGD(net.parameters(), lr=LR, momentum=0.1, nesterov=True)
    opt = optim.Adam(net.parameters(), lr=LR)
    adalr = AdaptiveLR(opt, LR, num_iterations=2000)

    num_epochs = 200
    init_epoch = 0

    for epoch in range(init_epoch, num_epochs):

        train(net, opt, adalr, batch_generator(pdb_loader(PDB_ZIP, TRAINING_SET, 100000, 19.99, 1.25), BATCH_SIZE), 100000, epoch)

        loss = evaluate(net, batch_generator(pdb_loader(PDB_ZIP, TRAINING_SET, 5000, 19.99, 1.25), BATCH_SIZE), 5000, (epoch+1)*1000)

        print("[Epoch %d/%d] (Validation Loss: %.5f" % (epoch + 1, num_epochs, loss))
