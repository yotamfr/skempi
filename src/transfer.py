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

USE_CUDA = True
BATCH_SIZE = 16
LR = 0.01


class SkempiReader(object):
    def __init__(self, producer, list_of_records, rotations, num_voxels=20):
        self.list_of_records = list_of_records
        self._p = np.random.permutation(len(list_of_records))
        self.nv = num_voxels
        self.rotations = rotations
        self.func = producer
        self.Q = deque()
        self.E = ThreadPoolExecutor(1)
        self.reset()

    def reset(self):
        self._pdb_ix = -1
        self.load_next_record()

    def load_next_record(self):
        self._pdb_ix += 1
        if self._pdb_ix == len(self._p):
            raise StopIteration
        if len(self.curr_record.mutations) > 1:
            self.load_next_record()

    @property
    def curr_record(self):
        return self.list_of_records[self._p[self._pdb_ix]]

    def read(self):
        self.load_next_record()
        self.E.submit(self.func, self.Q, self.curr_record, self.rotations, self.nv)


def non_blocking_producer_ddg_v1(queue, record, rotations, nv=20):
    assert len(record.mutations) == 1
    mut = record.mutations[0]
    res1 = record.struct[mut.chain_id][mut.i]
    res2 = record.mutant[mut.chain_id][mut.i]
    atoms1 = select_atoms_in_sphere(record.struct.atoms, res1.ca.coord, nv)
    atoms2 = select_atoms_in_sphere(record.mutant.atoms, res2.ca.coord, nv)
    onehot = list(get_features(record).values())
    for rot in rotations:
        voxels1 = get_4channel_voxels_around_res(atoms1, res1, rot, nv=nv)
        voxels2 = get_4channel_voxels_around_res(atoms2, res2, rot, nv=nv)
        queue.appendleft([voxels1, voxels2, onehot, record.ddg])


def batch_generator(loader, batch_size=BATCH_SIZE):

    def prepare_batch(v1, v2, oh, ddg):
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
        v1, v2, oh, ddg = zip(*batch)
        yield prepare_batch(v1, v2, oh, ddg)


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
        )

        self.fc1 = nn.Sequential(
            nn.Linear(400, 40),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(66, 33),
        )
        self.out = nn.Sequential(
            nn.Linear(33, 1)
        )

    def compress(self, x):
        x = self.features(x)
        x = x.view(len(x), -1)
        x = self.compression(x)
        return x

    def forward(self, x1, x2, oh):
        x1 = self.compress(x1)
        x2 = self.compress(x2)
        x = torch.cat([x1, x2], 1)
        x = self.fc1(x)
        x = torch.cat([x, oh], 1)
        x = self.fc2(torch.tanh(x))
        x = self.out(x)
        return x.view(-1)


def get_loss(ddg_p, ddg_m, ddg):
    return torch.mean(torch.abs(0.5 * (ddg_p - ddg_m) - ddg)) + \
           torch.mean(torch.abs(ddg_p + ddg_m))
    # return torch.sqrt(torch.mean((0.5 * (ddg_p - ddg_m) - ddg).pow(2))) + \
    #        torch.sqrt(torch.mean((ddg_p + ddg_m).pow(2)))


def evaluate(model, batch_generator, length_xy):
    model.eval()
    pbar = tqdm(total=length_xy, desc="calculation...")
    err, i = 0, 0
    preds = np.zeros((length_xy,))
    truth = np.zeros((length_xy,))
    for i, (v1, v2, oh, ddg) in enumerate(batch_generator):
        batch_size = ddg.size(0)
        ddg_p = model(v1, v2, oh)
        ddg_m = model(v2, v1, -oh)
        loss = get_loss(ddg_p, ddg_m, ddg)
        err += loss.item()
        start = i * batch_size
        end = i * batch_size + len(ddg)
        preds[start:end] = ddg_p.data.cpu().numpy()
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

    for i, (v1, v2, oh, ddg) in enumerate(batch_generator):

        opt.zero_grad()
        ddg_p = model(v1, v2, oh)
        ddg_m = model(v2, v1, -oh)
        loss = get_loss(ddg_p, ddg_m, ddg)
        adalr.update(loss.item())
        err += loss.item()
        loss.backward()
        opt.step()
        lr, e = adalr.lr, err/(i + 1)
        pbar.set_description("Training Loss:%.6f, LR: %.6f" % (e, lr))
        pbar.update(len(ddg))

    pbar.close()


def add_arguments(parser):
    parser.add_argument('-r', '--resume', required=True, type=str, metavar='PATH',
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
    dfs_freeze(net.features)
    dfs_freeze(net.compression)
    weights, bias = list(net.compression.parameters())[-2:]
    # weights.requires_grad = True
    # bias.requires_grad = True
    opt = optim.SGD(net.parameters(), lr=LR, momentum=0.9, nesterov=True)
    # opt = optim.Adam(net.parameters(), lr=LR)

    if os.path.isfile(args.resume):
        print("=> loading checkpoint '%s'" % args.resume)
        checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage)
        init_epoch = checkpoint['epoch']
        loaded_dict = checkpoint['net']
        net.load_state_dict(loaded_dict, strict=False)
        # opt.load_state_dict(checkpoint['opt'])
    else:
        print("=> no checkpoint found at '%s'" % args.resume)
        exit(0)

    ckptpath = args.out_dir
    model_summary(net)

    non_blocking_producer = non_blocking_producer_ddg_v1
    get_loss = get_loss

    records = load_skempi(skempi_df, PDB_PATH, True)
    indx1 = skempi_df.Protein.isin(G1 + G2 + G3 + G4 + G5)
    indx2 = (skempi_df.num_muts == 1)
    training_set = np.asarray(records)[indx1 & indx2]
    validation_set = np.asarray(records)[~indx1 & indx2]

    reader_trn = SkempiReader(non_blocking_producer_ddg_v1, training_set, [None], num_voxels=20)
    loader_trn = PdbLoader(reader_trn, len(training_set))
    reader_val = SkempiReader(non_blocking_producer_ddg_v1, validation_set, [None], num_voxels=20)
    loader_val = PdbLoader(reader_val, len(validation_set))

    init_epoch = 0
    num_epochs = args.num_epochs

    adalr = AdaptiveLR(opt, LR, num_iterations=200)

    # Move models to GPU
    if USE_CUDA:
        net = net.cuda()
    if USE_CUDA and args.resume:
        optimizer_cuda(opt)

    for epoch in range(init_epoch, num_epochs):

        train(net, opt, adalr, batch_generator(loader_trn), len(loader_trn))

        if epoch < num_epochs - 1 and epoch % args.eval_every != 0:
            continue

        loss, preds, truth = evaluate(net, batch_generator(loader_val), len(loader_val))
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
