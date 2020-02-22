# coding: utf-8

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.init as init


# see: _netD in https://github.com/pytorch/examples/blob/master/dcgan/main.py
class Discriminator_I(nn.Module):
    def __init__(self, nc=3, ndf=64, ngpu=1):
        super(Discriminator_I, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 96 x 96
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 48 x 48
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 24 x 24
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 12 x 12
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 6 x 6
            nn.Conv2d(ndf * 8, 1, 6, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)


class Discriminator_V(nn.Module):
    def __init__(self, nc=3, ndf=64, T=16, ngpu=1):
        super(Discriminator_V, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x T x 96 x 96
            nn.Conv3d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x T/2 x 48 x 48
            nn.Conv3d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x T/4 x 24 x 24
            nn.Conv3d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x T/8 x 12 x 12
            nn.Conv3d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x T/16  x 6 x 6
            Flatten(),
            nn.Linear((ndf*8)*(T//16)*6*6, 1),
            nn.Sigmoid()
        )


    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)

class Inconv(nn.Module):
    def __init__(self, in_ch, out_ch, use_bias = False):
        super(Inconv, self).__init__()
        self.inconv = nn.Sequential(
            #nn.ReflectionPad2d(3),
            nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=use_bias),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.inconv(x)
        return x

class Outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Outconv, self).__init__()
        self.outconv = nn.Sequential(
            #nn.ReflectionPad2d(3),
            nn.Conv2d(in_ch, out_ch, kernel_size=6, padding=0),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.outconv(x)
        return x

class OutconvTrans(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutconvTrans, self).__init__()
        self.outconv = nn.Sequential(
            #nn.ReflectionPad2d(3),
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=6, padding=0),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.outconv(x)
        return x

class G_down(nn.Module):
    def __init__(self, in_ch, out_ch, norm_layer = nn.BatchNorm2d, use_bias = False):
        super(G_down, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=4,
                      stride=2, padding=1, bias=use_bias),
            norm_layer(out_ch),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.down(x)
        return x

class G_up(nn.Module):
    def __init__(self, in_ch, out_ch, norm_layer = nn.BatchNorm2d, use_bias = False):
        super(G_up, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch,
                               kernel_size=4, stride=2,
                               padding=1,
                               bias=use_bias),
            norm_layer(out_ch),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class G_encode(nn.Module):
    def __init__(self, nz = 60, nef = 64, norm_layer = nn.BatchNorm2d, use_bias = False):
        super(G_encode, self).__init__()
        self.inc = nn.Sequential(nn.Conv2d(nz, nef, kernel_size=1, padding=0,
                      bias=use_bias),
                    norm_layer(nef),
                    nn.ReLU(True))
        down = []
        n_downsampling = 3
        mult = 0
        for i in range(n_downsampling):
            mult = 2**i
            down += [nn.Conv2d(nef * mult, nef * mult * 2, kernel_size=1,
                      stride=2, padding=0, bias=use_bias),
                     norm_layer(nef * mult * 2),
                     nn.ReLU(True)]
        self.down = nn.Sequential(*down)
        mult *= 2
        self.outc = nn.Sequential(nn.Conv2d(nef * mult, nef * mult * 2, kernel_size=1, padding=0,
                      bias=use_bias),
                    norm_layer(nef * mult * 2),
                    nn.ReLU(True))
    def forward(self,x):
        # print('G_encode Input =', x.size())
        out = self.outc(self.down(self.inc(x)))
        # print('G_encode Output =', out.size())
        return out

# see: _netG in https://github.com/pytorch/examples/blob/master/dcgan/main.py
class Generator_I(nn.Module):
    def __init__(self, nc=3, ngf=64, nz=60, ngpu=1):
        super(Generator_I, self).__init__()
        self.ngpu = ngpu
        self.input_nc = nc*2 # two frames
        self.output_nc = nc
        self.ngf = ngf
        self.nz = nz

        # Downsampling
        self.inc = Inconv(self.input_nc, ngf)
        self.encode = G_encode(nz, ngf)
        down = []
        n_downsampling = 3
        mult = 0
        for i in range(n_downsampling):
            mult = 2**i
            down += [G_down(ngf * mult, ngf * mult * 2)]
        self.down = nn.Sequential(*down)

        mult *= 2
        self.outc = Outconv(mult * ngf, mult * ngf * 2) # [batch_size 1024 1 1]

        # Upsampling
        self.resnet = OutconvTrans(mult * ngf * 2, mult * ngf)
        n_upsampling = 3
        up = []
        for i in range(n_upsampling):
            mult = 2**(n_upsampling - i)
            up += [G_up(ngf * mult, ngf * mult // 2)]
        self.up = nn.Sequential(*up)

        self.out = nn.Sequential(nn.ConvTranspose2d(ngf, self.output_nc, kernel_size=4, stride=2, padding=1,
                      bias=False),
                    nn.Tanh()) # [batch_size, 3 96 96]


    def forward(self, x, z):
        size = x.size()
        # print ('size is',size)

        self.bs, self.nf, self.h, self.w = size[0], size[1], size[3], size[4]
        # print ('The shape of x is:', x.shape)
        # print ('The shape of z is:', z.shape)

        input_img = x.contiguous().view(self.bs, -1, self.h, self.w)
        input_z = z.contiguous().view(self.bs, -1, 1, 1)
        # print ('The shape of input_img is:', input_img.shape)
        # print ('The shape of input_z is:', input_z.shape)

        down_raw = self.outc(self.down(self.inc(input_img)))
        # print ('The shape of down_raw is:', down_raw.shape)
        down_z = self.encode(input_z)
        # print ('The shape of down_z is:', down_z.shape)
        down_img = down_raw + down_z
        # print ('The shape of down_img is:', down_img.shape)

        output = self.up(self.resnet(down_img))
        # print ('The shape of output is:', output.shape)
        output = self.out(output)
        # print ('The shape of output is:', output.shape)

        return output


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0, gpu=True):
        super(GRU, self).__init__()

        output_size      = input_size
        self._gpu        = gpu
        self.hidden_size = hidden_size

        # define layers
        self.gru    = nn.GRUCell(input_size, hidden_size)
        self.drop   = nn.Dropout(p=dropout)
        self.linear = nn.Linear(hidden_size, output_size)
        self.bn     = nn.BatchNorm1d(output_size, affine=False)

    def forward(self, inputs, n_frames):
        '''
        inputs.shape()   => (batch_size, input_size)
        outputs.shape() => (seq_len, batch_size, output_size)
        '''
        outputs = []
        for i in range(n_frames):
            self.hidden = self.gru(inputs, self.hidden)
            inputs = self.linear(self.hidden)
            outputs.append(inputs)
        outputs = [ self.bn(elm) for elm in outputs ]
        outputs = torch.stack(outputs)
        return outputs

    def initWeight(self, init_forget_bias=1):
        # See details in https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/rnn.py
        for name, params in self.named_parameters():
            if 'weight' in name:
                init.xavier_uniform(params)

            # initialize forget gate bias
            elif 'gru.bias_ih_l' in name:
                b_ir, b_iz, b_in = params.chunk(3, 0)
                init.constant(b_iz, init_forget_bias)
            elif 'gru.bias_hh_l' in name:
                b_hr, b_hz, b_hn = params.chunk(3, 0)
                init.constant(b_hz, init_forget_bias)
            else:
                init.constant(params, 0)

    def initHidden(self, batch_size):
        self.hidden = Variable(torch.zeros(batch_size, self.hidden_size))
        if self._gpu == True:
            self.hidden = self.hidden.cuda()


''' utils '''

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
