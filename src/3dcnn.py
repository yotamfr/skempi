import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim

from tempfile import gettempdir
from collections import OrderedDict
from scipy.stats import pearsonr
import pickle

try:
    from src.pytorch_utils import *
    from src.skempi_utils import *
    from src.grid_utils import *
except ImportError:
    from pytorch_utils import *
    from skempi_utils import *
    from grid_utils import *

USE_CUDA = True
BATCH_SIZE = 32
LR = 0.
NUM_DESCRIPTORS = 2
# NUM_DESCRIPTORS = 8

rotations_x = [rot_x(r * 2 * math.pi) for r in np.arange(0, .99, .25)]
rotations_y = [rot_y(r * 2 * math.pi) for r in np.arange(0, .99, .25)]
rotations_z = [rot_z(r * 2 * math.pi) for r in np.arange(0, .99, .25)]
augmentations = rotations_x + rotations_y + rotations_z


class SingleMutationLoader(object):

    def __init__(self, skempi_records, augmentations):
        self._records = []
        for record in skempi_records:
            struct = record.struct
            ddg = record.ddg
            group = record.group
            for mut in record.mutations:
                for rot in augmentations:
                    rec = SkempiRecord(struct, [mut], ddg, group)
                    self._records.append([rot, rec])
        self._l_cache = {}
        self._v_cache = {}
        self._curr = 0

    def reset(self):

        self._curr = 0

    def __iter__(self):
        return self

    def next(self):
        if self._curr < len(self._records):
            rot, rec = self._records[self._curr]
            assert len(rec.mutations) == 1
            mut = rec.mutations[0]
            struct = rec.struct
            res = struct[mut.chain_id][mut.i]

            try:
                view = self._v_cache[rec]
                view.rotate(rot)

            except KeyError:
                view = View(struct, res, num_voxels=20)
                self._v_cache[rec] = view

            data = view.voxels

            try:
                labels = self._l_cache[rec]

            except KeyError:
                bfactor = rec.get_bfactor()
                # hydphob = get_descriptor([mut], ARGP820101)
                # molweight = get_descriptor([mut], FASG760101)
                asa = rec.get_asa()
                # cp_a1, cp_b1, _ = rec.get_shells_cp(2.0, 4.0)
                # cp_a2, cp_b2, _ = rec.get_shells_cp(4.0, 6.0)
                # labels = [bfactor, hydphob, molweight, asa, cp_a1, cp_b1, cp_a2, cp_b2]
                labels = [bfactor, asa]

                self._l_cache[rec] = labels

            self._curr += 1

            return data, np.asarray(labels)
        else:
            raise StopIteration

    def __next__(self):
        self.next()

    def __str__(self):
        return "<Loader: %d>" % len(self)

    def __len__(self):
        return len(self._records)


def batch_generator(loader, batch_size=BATCH_SIZE):

    def prepare_batch(data, labels):
        dat_var = Variable(torch.FloatTensor(data))
        lbl_var = Variable(torch.FloatTensor(labels))
        if USE_CUDA:
            dat_var = dat_var.cuda()
            lbl_var = lbl_var.cuda()
        return dat_var, lbl_var

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
        data, labels = zip(*batch)
        yield prepare_batch(data, labels)


class MLP(nn.Module):

    def __init__(self, input_size=100, dropout=0.3):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 10),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(10, 1)
        )

    def forward(self, x):
        y = self.model(x)
        return y


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
        self.info = nn.Sequential(
            nn.Linear(10800, 1000),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1000, 100),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        for j in range(NUM_DESCRIPTORS):
            setattr(self, "r%d" % j, MLP())

    def forward(self, x):
        x = self.features(x)
        x = x.view(len(x), -1)
        x = self.info(x)
        x = [getattr(self, "r%d" % j)(x) for j in range(NUM_DESCRIPTORS)]
        x = torch.cat(x, 1)
        return x


def get_loss(y_hat, y):
    mse = nn.MSELoss().cuda()
    total_loss = 0
    for j in range(NUM_DESCRIPTORS):
        total_loss += mse(y_hat[:, j], y[:, j])
    return total_loss


def evaluate(model, batch_generator, length_xy, batch_size=BATCH_SIZE):
    model.eval()
    pbar = tqdm(total=length_xy, desc="calculation...")
    err, i = 0, 0
    preds = np.zeros((length_xy, NUM_DESCRIPTORS))
    truth = np.zeros((length_xy, NUM_DESCRIPTORS))
    for i, (x, y) in enumerate(batch_generator):
        y_hat = model(x)
        loss = get_loss(y_hat, y)
        err += loss.item()
        start = i * batch_size
        end = i * batch_size + len(x)
        preds[start:end, :] = y_hat.data.cpu().numpy()
        truth[start:end, :] = y.data.cpu().numpy()
        pbar.update(len(y))
    cors = [pearsonr(preds[:, j], truth[:, j])[0] for j in range(NUM_DESCRIPTORS)]
    desc = " ".join("r%d=%.2f" % (i, cor) for (i, cor) in enumerate(cors))
    pbar.set_description(desc)
    pbar.close()
    return err / (i + 1)


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
        pbar.set_description("Training Loss:%.4f, LR: %.4f" % (e, lr))
        pbar.update(len(y))

    pbar.close()


def add_arguments(parser):

    parser.add_argument('-r', '--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument("-e", "--eval_every", type=int, default=1,
                        help="How often to evaluate on the validation set.")
    parser.add_argument("--num_epochs", type=int, default=200,
                        help="How many epochs to train the model?")
    parser.add_argument("-o", "--out_dir", type=str, required=False,
                        default=gettempdir(), help="Specify the output directory.")
    parser.add_argument("-s", "--seed", type=int, default=9898,
                        help="Sets the seed for generating random number.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()

    skempi_structs = load_skempi_structs("../data/pdbs", compute_dist_mat=False, carbons_only=False)

    skempi_records = load_skempi_records(skempi_structs)

    training_set = [record for record in skempi_records if 1 <= record.group <= 4]
    validation_set = [record for record in skempi_records if record.group == 5]

    loader_trn = SingleMutationLoader(training_set, augmentations)
    loader_val = SingleMutationLoader(validation_set, augmentations)

    net = CNN3dV1()
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

    adalr = AdaptiveLR(opt, LR, num_iterations=1000)

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
