import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim

from tempfile import gettempdir
from scipy.stats import pearsonr

from pymongo import MongoClient
from bson.binary import Binary
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
LR = 0.01

rotations_x = [rot_x(r * 2 * math.pi) for r in np.arange(0, .99, .25)]
rotations_y = [rot_y(r * 2 * math.pi) for r in np.arange(0, .99, .25)]
rotations_z = [rot_z(r * 2 * math.pi) for r in np.arange(0, .99, .25)]
augmentations = rotations_x + rotations_y + rotations_z


class SingleMutationLoader(object):

    def __init__(self, dataframe, augmentations, vcache):
        self._records = []
        self._curr = 0
        for _, row in dataframe.iterrows():
            for rot in augmentations:
                self._records.append([rot, row])
        self._vcache = vcache

    def reset(self):
        self._curr = 0

    def __iter__(self):
        return self

    def next(self):
        if self._curr < len(self._records):
            rot, row = self._records[self._curr]
            try:
                voxels1, voxels2, ddg = self._vcache[self._curr]
            except KeyError:
                rec1 = skempi_record_from_row(row)
                rec2 = reversed(rec1)
                assert rec1.ddg == -rec2.ddg
                assert len(rec1.mutations) == 1
                assert len(rec2.mutations) == 1
                mut1 = rec1.mutations[0]
                struct1 = rec1.struct
                res1 = struct1[mut1.chain_id][mut1.i]
                mut2 = rec2.mutations[0]
                struct2 = rec2.struct
                res2 = struct2[mut2.chain_id][mut2.i]
                assert mut1.i == mut2.i
                voxels1 = get_4channel_voxels(struct1, res1, rot, 20)
                voxels2 = get_4channel_voxels(struct2, res2, rot, 20)
                ddg = rec1.ddg
                self._vcache[self._curr] = (voxels1, voxels2, ddg)
            self._curr += 1
            return voxels1, voxels2, [ddg]
        else:
            raise StopIteration

    def __next__(self):
        self.next()

    def __str__(self):
        return "<SingleMutationLoader: %d>" % len(self)

    def __len__(self):
        return len(self._records)


def batch_generator(loader, batch_size=BATCH_SIZE):

    def prepare_batch(data1, data2, labels):
        dat_var1 = Variable(torch.FloatTensor(data1))
        dat_var2 = Variable(torch.FloatTensor(data2))
        lbl_var = Variable(torch.FloatTensor(labels))
        if USE_CUDA:
            dat_var1 = dat_var1.cuda()
            dat_var2 = dat_var2.cuda()
            lbl_var = lbl_var.cuda()
        return dat_var1, dat_var2, lbl_var
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
        dat1, dat2, lbl = zip(*batch)
        yield prepare_batch(dat1, dat2, lbl)


class CNN3dV1(nn.Module):    # https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-017-1702-0#Sec28

    def __init__(self, dropout=0.1):
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
        self.regressor = nn.Sequential(
            nn.Linear(100, 10),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(10, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(len(x), -1)
        x = self.info(x)
        x = self.regressor(x)
        return x


def get_loss(y_hat, y):
    loss = (y - y_hat).pow(2).sum() / y.size(0)
    return loss


def evaluate(model, batch_generator, length_xy, batch_size=BATCH_SIZE):
    model.eval()
    pbar = tqdm(total=length_xy, desc="calculation...")
    err, i = 0, 0
    preds = np.zeros((length_xy, 1))
    truth = np.zeros((length_xy, 1))
    for i, (x1, x2, y) in enumerate(batch_generator):
        y1 = model(x1)
        y2 = model(x2)
        y_hat = y1 - y2
        loss = get_loss(y_hat, y)
        err += loss.item()
        start = i * batch_size
        end = i * batch_size + len(y)
        preds[start:end, :] = y_hat.data.cpu().numpy()
        truth[start:end, :] = y.data.cpu().numpy()
        pbar.update(len(y))
        cor, _ = pearsonr(preds[:end, :], truth[:end, :])
        e = err / (i + 1)
        pbar.set_description("Validation Loss:%.4f, Perasonr: %.4f" % (e, cor))
    pbar.close()
    return e, cor


def train(model, opt, adalr, batch_generator, length_xy):

    model.train()

    pbar = tqdm(total=length_xy, desc="calculating...")

    err = 0

    for i, (x1, x2, y) in enumerate(batch_generator):

        opt.zero_grad()
        y1 = model(x1)
        y2 = model(x2)
        y_hat = y1 - y2
        loss = get_loss(y_hat, y)
        adalr.update(loss.item())
        err += loss.item()
        loss.backward()
        opt.step()
        lr, e = adalr.lr, err/(i + 1)
        pbar.set_description("Training Loss:%.4f, LR: %.4f" % (e, lr))
        pbar.update(len(y))

    pbar.close()


def serialize(obj):
    return Binary(pickle.dumps(obj, protocol=2))


def deserialize(bin_obj):
    return pickle.loads(bin_obj)


class Cache(object):

    def __init__(self, collection, serialize=lambda x: x, deserialize=lambda x: x):
        self.db = collection
        self.serialize = serialize
        self.deserialize = deserialize

    def __setitem__(self, key, item):
        self.db.update_one({"_id": hash(key)}, {"$set": {"data": self.serialize(item)}}, upsert=True)

    def __getitem__(self, key):
        load = self.deserialize
        c = self.db.count({"_id": hash(key)})
        if c == 0:
            raise KeyError(key)
        elif c == 1:
            return load(self.db.find_one({"_id": hash(key)})["data"])
        else:
            raise ValueError("found more than one result for: '%s'" % key)


def add_arguments(parser):
    parser.add_argument("--mongo_url", type=str, default='mongodb://localhost:27017/',
                        help="Supply the URL of MongoDB")
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
    parser.add_argument("-d", "--device", type=str, choices=["0", "1", "2", "3"],
                        default="0", help="Choose a device to run on.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    df = skempi_df[[len(parse_mutations(s)) == 1 for s in skempi_df["Mutation(s)_cleaned"]]]
    training_set = df[[1 <= skempi_group_from_row(row) <= 4 for _, row in df.iterrows()]]
    validation_set = df[[skempi_group_from_row(row) == 5 for _, row in df.iterrows()]]

    client = MongoClient(args.mongo_url)
    cache_trn = Cache(client['skempi']['siamese_train'], serialize=serialize, deserialize=deserialize)
    cache_val = Cache(client['skempi']['siamese_valid'], serialize=serialize, deserialize=deserialize)

    loader_trn = SingleMutationLoader(training_set, augmentations, vcache=cache_trn)
    loader_val = SingleMutationLoader(validation_set, [None], vcache=cache_val)

    net = CNN3dV1()
    opt = optim.SGD(net.parameters(), lr=LR, momentum=0.9, nesterov=True)

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

        loss, _ = evaluate(net, batch_generator(loader_val), len(loader_val))

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
