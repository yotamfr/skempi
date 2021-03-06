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
from grid_utils import *

USE_CUDA = True
BATCH_SIZE = 32
LR = 0.0001


def batch_generator(loader, batch_size=BATCH_SIZE):

    def prepare_batch(data1, data2):
        dat_var1 = Variable(torch.FloatTensor(data1))
        dat_var2 = Variable(torch.FloatTensor(data2))
        if USE_CUDA:
            dat_var1 = dat_var1.cuda()
            dat_var2 = dat_var2.cuda()
        return dat_var1, dat_var2
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
        inputs, labels = zip(*batch)
        yield prepare_batch(inputs, labels)


class CNN3dV1(nn.Module):    # https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-017-1702-0#Sec28

    def __init__(self, dropout=0.3):
        super(CNN3dV1, self).__init__()

        self.features = nn.Sequential(
            nn.Conv3d(4, 100, kernel_size=(3, 3, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv3d(100, 200, kernel_size=(3, 3, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.MaxPool3d((2, 2, 2)),
            nn.Conv3d(200, 400, kernel_size=(3, 3, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.MaxPool3d((2, 2, 2)),
        )
        self.classification = nn.Sequential(
            nn.Linear(10800, 1000),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1000, 100),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(100, 20)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(len(x), -1)
        x = self.classification(x)
        x = F.softmax(x, 1)
        return x


class CNN3dV2(nn.Module):

    def __init__(self, dropout=0.3):
        super(CNN3dV2, self).__init__()

        self.features = nn.Sequential(
            nn.Conv3d(4, 100, kernel_size=(3, 3, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv3d(100, 200, kernel_size=(3, 3, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.MaxPool3d((2, 2, 2)),
            nn.Conv3d(200, 400, kernel_size=(3, 3, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.MaxPool3d((2, 2, 2)),
        )
        self.compression = nn.Sequential(
            nn.Linear(10800, 1000),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1000, 200),
            nn.ReLU(inplace=True)
        )
        self.regression = nn.Sequential(
            nn.Linear(200, 40)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(len(x), -1)
        x = self.compression(x)
        x = self.regression(x)
        return x


class CNN3dV3(nn.Module):

    def __init__(self, dropout=0.3):
        super(CNN3dV3, self).__init__()

        self.features = nn.Sequential(
            nn.Conv3d(4, 100, kernel_size=(3, 3, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv3d(100, 200, kernel_size=(3, 3, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.MaxPool3d((2, 2, 2)),
            nn.Conv3d(200, 400, kernel_size=(3, 3, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.MaxPool3d((2, 2, 2)),
        )
        self.classification = nn.Sequential(
            nn.Linear(10800, 1000),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1000, 60)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(len(x), -1)
        x = self.classification(x)
        return x


# class CNN3dV4(nn.Module):
#
#     def __init__(self, dropout=0.3):
#         super(CNN3dV4, self).__init__()
#
#         self.features1 = nn.Sequential(
#             nn.Conv3d(4, 20, kernel_size=(3, 3, 3)),
#             nn.ReLU(inplace=True),
#             nn.Dropout(dropout)
#         )
#         self.features2 = nn.Sequential(
#             nn.Conv3d(20, 100, kernel_size=(3, 3, 3)),
#             nn.ReLU(inplace=True),
#             nn.Dropout(dropout),
#             nn.Conv3d(100, 200, kernel_size=(3, 3, 3)),
#             nn.ReLU(inplace=True),
#             nn.Dropout(dropout),
#             nn.MaxPool3d((2, 2, 2)),
#             nn.Conv3d(200, 400, kernel_size=(3, 3, 3)),
#             nn.ReLU(inplace=True),
#             nn.Dropout(dropout),
#             nn.MaxPool3d((2, 2, 2))
#         )
#
#         self.classification = nn.Sequential(
#             nn.Linear(540, 20)
#         )
#
#         self.regression = nn.Sequential(
#             nn.Linear(10800, 1000),
#             nn.ReLU(inplace=True),
#             nn.Dropout(dropout),
#             nn.Linear(1000, 40)
#         )
#
#     def forward(self, x):
#         x = self.features1(x)
#         y1 = x.view(len(x), -1)
#         x = self.features2(x)
#         x = x.view(len(x), -1)
#         x = self.classification(x)
#         return x


def get_loss_v1(y_hat, y):
    if USE_CUDA:
        return nn.BCELoss().cuda()(y_hat, y)
    else:
        return nn.BCELoss()(y_hat, y)


def get_loss_v2(y_hat, y):
    if USE_CUDA:
        return nn.MSELoss().cuda()(y_hat, y)
    else:
        return nn.MSELoss()(y_hat, y)


def get_loss_v3(y_hat, y):
    if USE_CUDA:
        bce = nn.BCELoss().cuda()(F.softmax(y_hat[:, :20]), y[:, :20])
        mse = nn.MSELoss().cuda()(y_hat[:, 20:], y[:, 20:])
    else:
        bce = nn.BCELoss()(F.softmax(y_hat[:, :20]), y[:, :20])
        mse = nn.MSELoss()(y_hat[:, 20:], y[:, 20:])
    return bce + mse


def evaluate(model, batch_generator, length_xy):
    model.eval()
    pbar = tqdm(total=length_xy, desc="calculation...")
    err, i = 0, 0
    preds = np.zeros((length_xy, 40))
    truth = np.zeros((length_xy, 40))
    for i, (x, y) in enumerate(batch_generator):
        batch_size = x.size(0)
        y_hat = model(x)
        loss = get_loss(y_hat, y)
        err += loss.item()
        start = i * batch_size
        end = i * batch_size + len(y)
        preds[start:end, :] = y_hat.data.cpu().numpy()
        truth[start:end, :] = y.data.cpu().numpy()
        pbar.update(len(y))
        e = err / (i + 1)
        pbar.set_description("Validation Loss:%.6f" % (e,))
    pbar.close()
    return e


def train(model, opt, adalr, batch_generator, length_xy):

    model.train()

    pbar = tqdm(total=length_xy, desc="calculating...")

    err = 0

    for i, (x, y) in enumerate(batch_generator):

        opt.zero_grad()
        y_hat = model(x)
        loss = get_loss(y_hat, y)
        adalr.update(loss.item())
        err += loss.item()
        loss.backward()
        opt.step()
        lr, e = adalr.lr, err/(i + 1)
        pbar.set_description("Training Loss:%.6f, LR: %.6f" % (e, lr))
        pbar.update(len(y))

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

    net = CNN3dV2()
    opt = optim.Adam(net.parameters(), lr=LR)

    ckptpath = args.out_dir
    model_summary(net)

    non_blocking_producer = non_blocking_producer_v2
    get_loss = get_loss_v2

    rotations = get_xyz_rotations(.5)
    reader_trn = PdbReader(non_blocking_producer, TRAINING_SET, rotations)
    loader_trn = PdbLoader(reader_trn, 10000, len(rotations))
    reader_val = PdbReader(non_blocking_producer, VALIDATION_SET)
    loader_val = PdbLoader(reader_val, 10000)

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

    adalr = AdaptiveLR(opt, LR, num_iterations=10000)

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
            print("[Epoch %d/%d] (Validation Loss: %.6f" % (epoch + 1, num_epochs, loss))

        save_checkpoint({
            'lr': adalr.lr,
            'epoch': epoch,
            'net': net.state_dict(),
            'opt': opt.state_dict()
        }, loss, "skempi", ckptpath)

        loader_val.reset()
        loader_trn.reset()
