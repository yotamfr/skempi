import torch
import torch.nn as nn
import torch.nn.modules.conv as conv
from torch_utils import *


class AddCoords(nn.Module):
    def __init__(self, radius_channel=True):
        super(AddCoords, self).__init__()
        self.radius_channel = radius_channel

    def forward(self, in_tensor):
        """
        in_tensor: (batch_size, channels, x_dim, y_dim)
        [0,0,0,0]   [0,1,2,3]   [0,0,0,0]
        [1,1,1,1]   [0,1,2,3]   [0,0,0,0]  << (i,j,k)th coordinates of pixels added as separate channels
        [2,2,2,2]   [0,1,2,3]   [0,0,0,0]
        taken from mkocabas.
        """
        batch_size, _, dim_z, dim_y, dim_x = in_tensor.shape
        xx_ones = torch.ones([1, 1, 1, 1, dim_x], dtype=torch.int32, device=device)
        yy_ones = torch.ones([1, 1, 1, 1, dim_y], dtype=torch.int32, device=device)

        xy_range = torch.arange(dim_y, dtype=torch.int32, device=device)[None, None, None, :, None]
        yx_range = torch.arange(dim_x, dtype=torch.int32, device=device)[None, None, None, :, None]
        zz_range = torch.arange(dim_z, dtype=torch.int32, device=device)[None, None, None, :, None]

        xy_channel = torch.matmul(xy_range.float(), xx_ones.float())
        xx_channel = xy_channel.repeat(batch_size, 1, dim_z, 1, 1).permute(0, 1, 4, 3, 2)

        yx_channel = torch.matmul(yx_range.float(), yy_ones.float())
        yy_channel = yx_channel.repeat(batch_size, 1, dim_z, 1, 1).permute(0, 1, 3, 4, 2)

        zx_channel = torch.matmul(zz_range.float(), xx_ones.float())
        zz_channel = zx_channel.repeat(batch_size, 1, dim_y, 1, 1).permute(0, 1, 4, 2, 3)

        if dim_y > 1:
            xx_channel = xx_channel / (dim_y - 1)
            xx_channel = xx_channel * 2 - 1

        if dim_x > 1:
            yy_channel = yy_channel / (dim_x - 1)
            yy_channel = yy_channel * 2 - 1

        if dim_z > 1:
            zz_channel = zz_channel / (dim_z - 1)
            zz_channel = zz_channel * 2 - 1

        out = torch.cat([in_tensor, xx_channel, yy_channel, zz_channel], dim=1)

        if self.radius_channel:
            radius_calc = torch.sqrt(torch.pow(xx_channel, 2) + torch.pow(yy_channel, 2) + torch.pow(zz_channel, 2))
            out = torch.cat([out, radius_calc], dim=1).cuda()

        return out


class CoordConv3d(nn.Module):
    """ add any additional coordinate channels to the input tensor """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, radius_channel=True):
        super(CoordConv3d, self).__init__()
        self.addcoord = AddCoords(radius_channel=radius_channel)
        self.conv = nn.Conv3d(in_channels + 3 + int(radius_channel), out_channels,
                              kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, in_tensor):
        out = self.addcoord(in_tensor)
        out = self.conv(out)
        return out


class CoordConvTranspose3d(nn.Module):
    """CoordConvTranspose layer for segmentation tasks."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True, dilation=1, radius_channel=True):
        super(CoordConvTranspose3d, self).__init__()
        self.addcoord = AddCoords(radius_channel=radius_channel)
        self.convT = nn.ConvTranspose3d(in_channels + 3 + int(radius_channel), out_channels,
                                        kernel_size, stride, padding, output_padding, groups, bias, dilation)

    def forward(self, in_tensor):
        out = self.addcoord(in_tensor)
        out = self.convT(out)
        return out
