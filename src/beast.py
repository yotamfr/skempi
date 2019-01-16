import torch
from torch import nn
from torch import optim

from vae import *
from loader import *
from skempi_lib import *
from torch_utils import *

BATCH_SIZE = 32
LR = 1e-3


def get_loss(aa_hat, aa, x_hat, x, mu, logvar):
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    L1 = nn.L1Loss(reduction='sum')(x_hat, x)
    # BCE = nn.BCELoss(reduction='sum')(x_hat, x)
    CE = -F.log_softmax(aa_hat, 1).gather(1, aa.unsqueeze(1)).mean()
    return L1, KLD, CE


def train(model, opt, batch_generator, length_xy, n_iter):

    model.train()
    pbar = tqdm(total=length_xy, desc="calculating...")
    err = 0.

    for i, (aa, x, y) in enumerate(batch_generator):
        opt.zero_grad()
        aa_hat, y_hat, mu, logvar = model(x)
        recons, kld, ce = get_loss(aa_hat, aa, y_hat, y, mu, logvar)
        # loss = recons + kld + ce
        loss = ce

        writer.add_scalars('VAE/Loss', {"train": loss.item()}, n_iter)
        writer.add_scalars('VAE/Recons', {"train": recons.item()}, n_iter)
        writer.add_scalars('VAE/KLD', {"train": kld.item()}, n_iter)
        writer.add_scalars('VAE/CE', {"train": ce.item()}, n_iter)

        n_iter += 1
        err += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25, norm_type=2)
        loss.backward()
        opt.step_and_update_lr(loss.item())
        lr, e = opt.lr, err/(i + 1)
        pbar.set_description("Training Loss:%.5f, LR: %.5f" % (e, lr))
        pbar.update(len(y))

    pbar.close()
    return n_iter


def evaluate(model, batch_generator, length_xy, n_iter):

    model.eval()
    pbar = tqdm(total=length_xy, desc="calculation...")
    err, loss1, loss2, loss3 = 0., 0., 0., 0.

    for i, (aa, x, y) in enumerate(batch_generator):
        aa_hat, y_hat, mu, logvar = model(x)
        recons, kld, ce = get_loss(aa_hat, aa, y_hat, y, mu, logvar)
        # loss = recons + kld + ce
        loss = ce

        err += loss.item()
        loss1 += recons.item()
        loss2 += kld.item()
        loss3 += ce.item()

        pbar.set_description("Validation Loss:%.5f" % (err/(i + 1),))
        pbar.update(len(y))

    writer.add_scalars('VAE/Loss', {"valid": err/(i + 1)}, n_iter)
    writer.add_scalars('VAE/Recons', {"valid": loss1/(i + 1)}, n_iter)
    writer.add_scalars('VAE/KLD', {"valid": loss2/(i + 1)}, n_iter)
    writer.add_scalars('VAE/CE', {"valid": loss3/(i + 1)}, n_iter)

    pbar.close()
    return err/(i + 1)


def add_arguments(parser):
    parser.add_argument('-r', '--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument("-e", "--eval_every", type=int, default=1,
                        help="How often to evaluate on the validation set.")
    parser.add_argument('-n', "--num_epochs", type=int, default=200,
                        help="How many epochs to train the model?")
    parser.add_argument("-o", "--out_dir", type=str, required=False,
                        default=gettempdir(), help="Specify the output directory.")


def handle_error(pdb, err):
    print(pdb, str(err))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()

    net = VAE2(nc=24, ngf=64, ndf=64, latent_variable_size=256)
    net.to(device)
    # opt = optim.SGD(net.parameters(), lr=LR, momentum=0.9, nesterov=True)
    opt = ScheduledOptimizer(optim.Adam(net.parameters(), lr=LR), LR, num_iterations=200)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '%s'" % args.resume)
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage)
            init_epoch = checkpoint['epoch']
            net.load_state_dict(checkpoint['net'])
            opt.load_state_dict(checkpoint['opt'])
        else:
            print("=> no checkpoint found at '%s'" % args.resume)

    num_epochs = args.num_epochs
    init_epoch = 0
    n_iter = 0
    for epoch in range(init_epoch, num_epochs):

        train_iter, eval_iter = 20000, 5000
        loader = pdb_loader(PDB_ZIP, TRAINING_SET, train_iter, 19.9, 1.25, handle_error=handle_error)
        n_iter = train(net, opt, batch_generator(loader, BATCH_SIZE), train_iter, n_iter)

        if epoch < num_epochs - 1 and epoch % args.eval_every != 0:
            continue

        loader = pdb_loader(PDB_ZIP, VALIDATION_SET, eval_iter, 19.9, 1.25, handle_error=handle_error)
        loss = evaluate(net, batch_generator(loader, BATCH_SIZE), eval_iter, n_iter)

        print("[Epoch %d/%d] (Validation Loss: %.5f" % (epoch + 1, num_epochs, loss))

        save_checkpoint({
            'lr': opt.lr,
            'epoch': epoch,
            'net': net.state_dict(),
            'opt': opt.state_dict()
        }, loss, "beast", args.out_dir)
