import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim

from tempfile import gettempdir
from scipy.stats import pearsonr
from tqdm import tqdm

from torch_utils import *
from reader_utils import *
from skempi_lib import *
from grid_utils import *

VERBOSE = True
USE_CUDA = True
BATCH_SIZE = 4
LR = 0.01


def prepare_siamese_batch(batch):
    v1, v2, oh, ddg = zip(*batch)
    if USE_CUDA:
        dat_var1 = Variable(torch.FloatTensor(v1)).cuda()
        dat_var2 = Variable(torch.FloatTensor(v2)).cuda()
        ddg_var = Variable(torch.FloatTensor(ddg)).cuda()
        oh_var = Variable(torch.FloatTensor(oh)).cuda()
    else:
        dat_var1 = Variable(torch.FloatTensor(v1))
        dat_var2 = Variable(torch.FloatTensor(v2))
        ddg_var = Variable(torch.FloatTensor(ddg))
        oh_var = Variable(torch.FloatTensor(oh))
    return dat_var1, dat_var2, oh_var, ddg_var


def prepare_batch(batch):
    vox, oh, ddg = zip(*batch)
    if USE_CUDA:
        dat_var = Variable(torch.FloatTensor(vox)).cuda()
        ddg_var = Variable(torch.FloatTensor(ddg)).cuda()
        oh_var = Variable(torch.FloatTensor(oh)).cuda()
    else:
        dat_var = Variable(torch.FloatTensor(vox))
        ddg_var = Variable(torch.FloatTensor(ddg))
        oh_var = Variable(torch.FloatTensor(oh))
    return dat_var, oh_var, ddg_var


def batch_generator(loader, prepare_batch, batch_size=BATCH_SIZE):

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
        yield prepare_batch(batch)


class CNN3d(nn.Module):

    def __init__(self, dropout=0.2):
        super(CNN3d, self).__init__()

        self.features = nn.Sequential(
            nn.Conv3d(8, 64, kernel_size=(5, 5, 5)),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((2, 2, 2)),
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3)),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((2, 2, 2)),
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3)),
            nn.ReLU(inplace=True),
        )
        self.compression = nn.Sequential(
            nn.Linear(256, 100),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.fc = nn.Sequential(
            nn.Linear(120, 1),
            nn.Dropout(dropout),
        )

    def compress(self, x):
        x = self.features(x)
        x = x.view(len(x), -1)
        x = self.compression(x)
        return x

    def forward(self, x, oh):
        x = self.compress(x)
        x = torch.cat([x, oh], 1)
        x = self.fc(x)
        return x.view(-1)


def eval_model(model, variables):
    vox, oh, ddg = variables
    ddg = ddg.view(-1)
    ddg_hat = model(vox, oh)
    loss = nn.MSELoss().cuda()(ddg_hat, ddg) if USE_CUDA else nn.MSELoss()(ddg_hat, ddg)
    return loss, ddg, ddg_hat


def evaluate(model, batch_generator, length_xy, batch_size=BATCH_SIZE):
    model.eval()
    pbar = tqdm(total=length_xy, desc="calculation...")
    err, i = 0, 0
    preds = np.zeros((length_xy,))
    truth = np.zeros((length_xy,))
    for i, vars in enumerate(batch_generator):
        loss, ddg, ddg_hat = eval_model(model, vars)
        err += loss.item()
        start = i * batch_size
        end = i * batch_size + len(ddg)
        preds[start:end] = ddg_hat.data.cpu().numpy()
        truth[start:end] = ddg.data.cpu().numpy()
        pbar.update(len(ddg))
        e = err / (i + 1)
        pbar.set_description("Validation Loss:%.6f" % (e,))
    pbar.close()
    return e, preds, truth


def train(model, opt, adalr, batch_generator, length_xy):
    model.train()
    pbar = tqdm(total=length_xy, desc="calculating...")
    err = 0
    for i, vars in enumerate(batch_generator):
        opt.zero_grad()
        loss, ddg, ddg_hat = eval_model(model, vars)
        adalr.update(loss.item())
        err += loss.item()
        loss.backward()
        opt.step()
        lr, e = adalr.lr, err/(i + 1)
        pbar.set_description("Training Loss:%.6f, LR: %.6f" % (e, lr))
        pbar.update(len(ddg))
    pbar.close()


def add_arguments(parser):
    parser.add_argument('-r', '--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument("-e", "--eval_every", type=int, default=1,
                        help="How often to evaluate on the validation set.")
    parser.add_argument('-n', "--num_epochs", type=int, default=20,
                        help="How many epochs to train the model?")
    parser.add_argument("-o", "--out_dir", type=str, required=False,
                        default=gettempdir(), help="Specify the output directory.")
    parser.add_argument("-s", "--seed", type=int, default=9898,
                        help="Sets the seed for generating random number.")
    parser.add_argument("-d", "--device", type=str, choices=["0", "1", "2", "3"],
                        default="3", help="Choose a device to run on.")
    parser.add_argument("-g", '--debug', action='store_true', default=True,
                        help="Run in debug mode.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    net = CNN3d()

    # opt = optim.SGD(net.parameters(), lr=LR, momentum=0.9, nesterov=True)
    opt = optim.Adam(net.parameters(), lr=LR)

    if os.path.isfile(args.resume):
        print("=> loading checkpoint '%s'" % args.resume)
        checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage)
        init_epoch = checkpoint['epoch']
        loaded_dict = checkpoint['net']
        net.load_state_dict(loaded_dict, strict=False)
        opt.load_state_dict(checkpoint['opt'])
    else:
        print("=> no checkpoint found at '%s'" % args.resume)

    ckptpath = args.out_dir
    model_summary(net)

    non_blocking_producer = non_blocking_interface_producer

    indx1 = skempi_df.Protein.isin(G1 + G2 + G3 + G4 + G5)
    indx2 = (skempi_df.num_muts == 1)
    training_set = skempi_df.Protein[indx1 & indx2]
    validation_set = skempi_df.Protein[~indx1 & indx2]
    records = load_skempi(skempi_df[indx1].reset_index(), PDB_PATH, True, False)
    ddg_mapper = DDGMapper(records)
    rotations = get_xyz_rotations(0.5)
    # rotations = [None]
    reader_trn = SkempiReader(non_blocking_producer, training_set, ddg_mapper, rotations, num_voxels=20)
    loader_trn = PdbLoader(reader_trn, 2000, 19 * len(rotations))
    reader_val = SkempiReader(non_blocking_producer, validation_set, ddg_mapper, [None], num_voxels=20)
    loader_val = PdbLoader(reader_val, len(validation_set), 19)

    init_epoch = 0
    num_epochs = args.num_epochs

    adalr = AdaptiveLR(opt, LR, num_iterations=2000)

    # Move models to GPU
    if USE_CUDA:
        net = net.cuda()
    if USE_CUDA and args.resume:
        optimizer_cuda(opt)

    for epoch in range(init_epoch, num_epochs):

        train(net, opt, adalr, batch_generator(loader_trn, prepare_batch), len(loader_trn))

        if epoch < num_epochs - 1 and epoch % args.eval_every != 0:
            continue

        loss, preds, truth = evaluate(net, batch_generator(loader_val, prepare_batch), len(loader_val))
        cor, _ = pearsonr(preds, truth)

        if VERBOSE:
            print("[Epoch %d/%d] (Validation Loss: %.6f, PCC: %.6f" %
                  (epoch + 1, num_epochs, loss, cor))

        save_checkpoint({
            'lr': adalr.lr,
            'epoch': epoch,
            'net': net.state_dict(),
            'opt': opt.state_dict()
        }, loss, "skempi", ckptpath)

        loader_val.reset()
        loader_trn.reset()
