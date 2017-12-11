import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class GCN(nn.Module):
    '''
    The Global Convolutional Network module using large 1D
    Kx1 and 1xK kernels to represent 2D kernels
    '''
    def __init__(self, inplanes, planes, ks=7):
        super(GCN, self).__init__()
        self.conv_l1 = nn.Conv2d(inplanes, planes, kernel_size=(ks, 1),
                                 padding=(ks//2, 0))

        self.conv_l2 = nn.Conv2d(planes, planes, kernel_size=(1, ks),
                                 padding=(0, ks//2))
        self.conv_r1 = nn.Conv2d(inplanes, planes, kernel_size=(1, ks),
                                 padding=(0, ks//2))
        self.conv_r2 = nn.Conv2d(planes, planes, kernel_size=(ks, 1),
                                 padding=(ks//2, 0))

    def forward(self, x):
        x_l = self.conv_l1(x)
        x_l = self.conv_l2(x_l)

        x_r = self.conv_r1(x)
        x_r = self.conv_r2(x_r)

        x = x_l + x_r

        return x


class Refine(nn.Module):
    '''
    Simple residual block to refine the details of the activation maps
    '''
    def __init__(self, planes):
        super(Refine, self).__init__()
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv2(x)

        out = residual + x
        return out


class FCN(nn.Module):
    '''
    2D FCN network with 3 input channels. The small decoder is built
    with the GCN and Refine modules.
    '''
    def __init__(self, nout=1):
        super(FCN, self).__init__()

        self.nout = nout

        resnet = models.resnet50(pretrained=True)

        self.conv1 = resnet.conv1
        self.bn0 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.gcn1 = GCN(2048, self.nout)
        self.gcn2 = GCN(1024, self.nout)
        self.gcn3 = GCN(512, self.nout)
        self.gcn4 = GCN(64, self.nout)
        self.gcn5 = GCN(64, self.nout)

        self.refine1 = Refine(self.nout)
        self.refine2 = Refine(self.nout)
        self.refine3 = Refine(self.nout)
        self.refine4 = Refine(self.nout)
        self.refine5 = Refine(self.nout)
        self.refine6 = Refine(self.nout)
        self.refine7 = Refine(self.nout)
        self.refine8 = Refine(self.nout)
        self.refine9 = Refine(self.nout)
        self.refine10 = Refine(self.nout)
        self.transformer = nn.Conv2d(256, 64, kernel_size=1)

    def _regresser(self, inplanes):
        return nn.Sequential(
            nn.Conv2d(inplanes, inplanes, 3, padding=1, bias=False),
            nn.BatchNorm2d(inplanes//2),
            nn.ReLU(inplace=True),
            nn.Dropout(.1),
            nn.Conv2d(inplanes//2, self.nout, 1),
        )

    def forward(self, x):
        input = x
        x = self.conv1(x)
        x = self.bn0(x)
        x = self.relu(x)
        conv_x = x
        x = self.maxpool(x)
        pool_x = x

        fm1 = self.layer1(x)
        fm2 = self.layer2(fm1)
        fm3 = self.layer3(fm2)
        fm4 = self.layer4(fm3)

        gcfm1 = self.refine1(self.gcn1(fm4))
        gcfm2 = self.refine2(self.gcn2(fm3))
        gcfm3 = self.refine3(self.gcn3(fm2))
        gcfm4 = self.refine4(self.gcn4(pool_x))
        gcfm5 = self.refine5(self.gcn5(conv_x))

        fs1 = self.refine6(F.upsample_bilinear(gcfm1, fm3.size()[2:]) + gcfm2)
        fs2 = self.refine7(F.upsample_bilinear(fs1, fm2.size()[2:]) + gcfm3)
        fs3 = self.refine8(F.upsample_bilinear(fs2, pool_x.size()[2:]) + gcfm4)
        fs4 = self.refine9(F.upsample_bilinear(fs3, conv_x.size()[2:]) + gcfm5)
        out = self.refine10(F.upsample_bilinear(fs4, input.size()[2:]))

        return out


class MCFCN(FCN):
    '''
    The multi-channel version of the 2D FCN module.
    Adds a projection layer to take arbitrary number of inputs
    '''
    def __init__(self, nin=3, nout=1):
        super(MCFCN, self).__init__(nout)

        self.init_proj = nn.Sequential(
            nn.Conv2d(nin, 3, 1, padding=0, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.init_proj(x)
        out = super(MCFCN, self).forward(x)
        return out
