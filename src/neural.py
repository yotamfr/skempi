import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim

import os
from os import path as osp

from skempi_utils import *
from pytorch_utils import *
from tensorboardX import SummaryWriter

writer = SummaryWriter('runs')
USE_CUDA = True


def pearsonr_torch(x, y):
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    xm = x.sub(mean_x)
    ym = y.sub(mean_y)
    r_num = xm.dot(ym)
    r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
    r_val = r_num / r_den
    return r_val


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        raise NotImplementedError

    def get_loss(self, X, y):
        raise NotImplementedError

    def fit(self, X, y, valid=None, prefix="anonym", epochs=250):
        l_rate = 0.01
        optimiser = torch.optim.Adam(self.parameters(), lr=l_rate)

        for epoch in range(epochs):
            epoch += 1

            self.train()
            optimiser.zero_grad()
            loss, cor = self.get_loss(X, y)
            # writer.add_histogram('%s/Hist' % (prefix,), self.predict(X), epoch)
            writer.add_scalars('%s/Loss' % (prefix,), {"train": loss.item()}, epoch)
            writer.add_scalars('%s/PCC' % (prefix,), {"train": cor.item()}, epoch)
            loss.backward()  # back props
            optimiser.step()  # update the parameters

            if valid is not None:
                self.eval()
                X_val, y_val = valid
                loss, cor = self.get_loss(X_val, y_val)

                writer.add_scalars('%s/Loss' % (prefix,), {"valid": loss.item()}, epoch)
                writer.add_scalars('%s/PCC' % (prefix,), {"valid": cor.item()}, epoch)

    def predict(self, X):
        self.eval()
        return self.forward(Variable(torch.FloatTensor(X)).cuda() if USE_CUDA
                            else Variable(torch.FloatTensor(X))).view(-1).cpu().data.numpy()


class LinearRegressionModel(Model):
    def __init__(self, input_dim, output_dim=1):
        super(LinearRegressionModel, self).__init__()
        self.model = nn.Linear(input_dim, output_dim)
        if USE_CUDA: self.model = self.model.cuda()

    def forward(self, x):
        out = self.model(x).view(-1)
        return out

    def get_loss(self, X, y):
        criterion = nn.MSELoss().cuda() if USE_CUDA else nn.MSELoss()
        inputs = Variable(torch.FloatTensor(X))
        labels = Variable(torch.FloatTensor(y))
        if USE_CUDA:
            inputs = inputs.cuda()
            labels = labels.cuda()
        outputs = self.forward(inputs).view(-1)
        loss = criterion(outputs, labels)
        return loss, pearsonr_torch(outputs, labels)


class MultiLayerModel(Model):
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super(MultiLayerModel, self).__init__()
        self.r1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh(),
        )
        self.r2 = nn.Sequential(
            nn.Linear(input_dim + 1, output_dim),
        )
        if USE_CUDA:
            self.r1 = self.r1.cuda()
            self.r2 = self.r2.cuda()

    def forward(self, x):
        o = self.r2(torch.cat([x, self.r1(x)], 1))
        return o.view(-1)

    def get_loss(self, X, y):
        inp = Variable(torch.FloatTensor(X))
        lbl = Variable(torch.FloatTensor(y))
        if USE_CUDA:
            inp = inp.cuda()
            lbl = lbl.cuda()

        y_hat_p = self.forward(inp).view(-1)
        y_hat_m = self.forward(-inp).view(-1)
        z_hat_p = self.r1(inp).view(-1)
        z_hat_m = self.r1(-inp).view(-1)

        mse = nn.MSELoss().cuda() if USE_CUDA else nn.MSELoss()

        completeness0 = mse(0.5 * (y_hat_p - y_hat_m), lbl)
        consistency0 = mse(-y_hat_p, y_hat_m)
        completeness2 = mse(0.5 * (z_hat_p - z_hat_m), torch.sign(lbl))
        consistency2 = mse(-z_hat_p, z_hat_m)

        loss = completeness0 + consistency0 + completeness2 + consistency2
        return loss, pearsonr_torch(y_hat_p, lbl)
