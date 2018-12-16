import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input, size=1024):
        return input.view(input.size(0), size, 1, 1, 1)


class VAE(nn.Module):
    def __init__(self, nc, ngf, ndf, latent_variable_size):
        super(VAE, self).__init__()

        self.nc = nc
        self.ngf = ngf
        self.ndf = ndf
        self.latent_variable_size = latent_variable_size

        # encoder
        self.e1 = nn.Conv3d(nc, ndf, 4, 2, 1)
        self.bn1 = nn.BatchNorm3d(ndf)

        self.e2 = nn.Conv3d(ndf, ndf*2, 4, 2, 1)
        self.bn2 = nn.BatchNorm3d(ndf*2)

        self.e3 = nn.Conv3d(ndf*2, ndf*4, 4, 2, 1)
        self.bn3 = nn.BatchNorm3d(ndf*4)

        self.e4 = nn.Conv3d(ndf*4, ndf*8, 4, 2, 1)
        self.bn4 = nn.BatchNorm3d(ndf*8)

        self.e5 = nn.Conv3d(ndf*8, ndf*8*2, 4, 2, 1)
        self.bn5 = nn.BatchNorm3d(ndf*8*2)

        self.fc1 = nn.Linear(ndf*8*2, latent_variable_size)
        self.fc2 = nn.Linear(ndf*8*2, latent_variable_size)

        # decoder
        self.d1 = nn.Linear(latent_variable_size, ngf*8*2)

        self.d2 = nn.ConvTranspose3d(ngf*8*2, ngf*8, 4, 2, 1)
        self.bn6 = nn.BatchNorm3d(ngf*8, 1.e-3)

        self.d3 = nn.ConvTranspose3d(ngf*8, ngf*4, 4, 2, 1)
        self.bn7 = nn.BatchNorm3d(ngf*4, 1.e-3)

        self.d4 = nn.ConvTranspose3d(ngf*4, ngf*2, 4, 2, 1)
        self.bn8 = nn.BatchNorm3d(ngf*2, 1.e-3)

        self.d5 = nn.ConvTranspose3d(ngf*2, ngf, 4, 2, 1)
        self.bn9 = nn.BatchNorm3d(ngf, 1.e-3)

        self.d6 = nn.ConvTranspose3d(ngf, nc, 4, 2, 1)

        self.flatten = Flatten()
        self.unflatten = UnFlatten()

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()

    def encode(self, x):
        h1 = self.leakyrelu(self.bn1(self.e1(x)))
        h2 = self.leakyrelu(self.bn2(self.e2(h1)))
        h3 = self.leakyrelu(self.bn3(self.e3(h2)))
        h4 = self.leakyrelu(self.bn4(self.e4(h3)))
        h5 = self.leakyrelu(self.bn5(self.e5(h4)))
        h5 = self.flatten(h5)
        return self.fc1(h5), self.fc2(h5)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h1 = self.relu(self.d1(z))
        h1 = self.unflatten(h1, self.ngf*8*2)
        h2 = self.leakyrelu(self.bn6(self.d2(h1)))
        h3 = self.leakyrelu(self.bn7(self.d3(h2)))
        h4 = self.leakyrelu(self.bn8(self.d4(h3)))
        h5 = self.leakyrelu(self.bn9(self.d5(h4)))
        return self.d6(h5)

    def get_latent_var(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        res = self.decode(z)
        return res, mu, logvar
