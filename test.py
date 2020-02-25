# coding: utf-8

import os
import argparse
import glob
import time
import math
import skvideo.io
import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable

from models import Discriminator_I, Discriminator_V, Generator_I, GRU


parser = argparse.ArgumentParser(description='Start testing MoCoGAN.....')
parser.add_argument('--cuda', type=int, default=1,
                     help='set -1 when you use cpu')
parser.add_argument('--ngpu', type=int, default=1,
                     help='set the number of gpu you use')
parser.add_argument('--batch_size', type=int, default=16,
                     help='set batch_size, default: 16')
parser.add_argument('--niter', type=int, default=16,
                     help='set num of iterations, default: 120000')
parser.add_argument('--nModel', type=int, default=120000,
                     help='set epoch of pre_trained model you want to load')
parser.add_argument('--nPrevImg', type=int, default=2,
                     help='set the number of images you use as condition')
parser.add_argument('--nFrameLoad', type=int, default=16,
                     help='set the number of frames you use')

args        = parser.parse_args()
cuda        = args.cuda
ngpu        = args.ngpu
batch_size  = args.batch_size
nPrevImg    = args.nPrevImg
nFrameLoad  = args.nFrameLoad
nModel      = args.nModel
niter       = args.niter


seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
if cuda == True:
    torch.cuda.set_device(0)


''' prepare dataset '''

current_path = os.path.dirname(__file__)
resized_path = os.path.join(current_path, 'resized_data')
files = glob.glob(resized_path+'/*')
videos = [ skvideo.io.vread(file) for file in files ]
# transpose each video to (nc, n_frames, img_size, img_size), and devide by 255
videos = [ video.transpose(3, 0, 1, 2) / 255.0 for video in videos ]


''' prepare video sampling '''

n_videos = len(videos)
T = nFrameLoad + nPrevImg

# for true video
def trim(video):
    start = np.random.randint(0, video.shape[1] - (T+1))
    end = start + T
    return video[:, start:end, :, :]

# for input noises to generate fake video
# note that noises are trimmed randomly from n_frames to T for efficiency
def trim_noise(noise):
    start = np.random.randint(0, noise.size(1) - (T+1))
    end = start + T
    return noise[:, start:end, :, :, :]

def random_choice():
    X = []
    for _ in range(batch_size):
        video = videos[np.random.randint(0, n_videos-1)]
        video = torch.Tensor(trim(video))
        X.append(video)
    X = torch.stack(X)
    return X

# video length distribution
video_lengths = [video.shape[1] for video in videos]


''' set models '''

img_size = 96
nc = 3
ndf = 64 # from dcgan
ngf = 64
d_E = 10
hidden_size = 100 # guess
d_C = 50
d_M = d_E
nz  = d_C + d_M
criterion = nn.BCELoss()

gen_i = Generator_I(nc, ngf, nz, ngpu=ngpu)
gru = GRU(d_E, hidden_size, gpu=cuda)
gru.initWeight()


''' prepare for test '''

label = torch.FloatTensor()

def save_video(fake_video, epoch):
    outputdata = fake_video * 255
    outputdata = outputdata.astype(np.uint8)
    dir_path = os.path.join(current_path, 'result_videos')
    file_path = os.path.join(dir_path, 'Video_epoch-%d.mp4' % epoch)
    skvideo.io.vwrite(file_path, outputdata)

trained_path = os.path.join(current_path, 'trained_models')

''' adjust to cuda '''

if cuda == True:
    gen_i.cuda()
    gru.cuda()
    criterion.cuda()
    label = label.cuda()

gen_i.load_state_dict(torch.load(trained_path + '/Generator_I_epoch-' + str(nModel) + '.model'))
gru.load_state_dict(torch.load(trained_path + '/GRU_epoch-' + str(nModel) + '.model'))


''' gen input noise for fake video '''

def gen_z(n_frames):
    z_C = Variable(torch.randn(batch_size, d_C))
    #  repeat z_C to (batch_size, n_frames, d_C)
    z_C = z_C.unsqueeze(1).repeat(1, n_frames, 1)
    eps = Variable(torch.randn(batch_size, d_E))
    if cuda == True:
        z_C, eps = z_C.cuda(), eps.cuda()

    gru.initHidden(batch_size)
    # notice that 1st dim of gru outputs is seq_len, 2nd is batch_size
    z_M = gru(eps, n_frames).transpose(1, 0)
    z = torch.cat((z_M, z_C), 2)  # z.size() => (batch_size, n_frames, nz)
    return z.view(batch_size, n_frames, nz, 1, 1)

''' gen input noise for fake image '''

def gen_zi():
    z_C = Variable(torch.randn(batch_size, d_C))
    z_C = z_C.unsqueeze(1)
    eps = Variable(torch.randn(batch_size, d_E))
    if cuda == True:
        z_C, eps = z_C.cuda(), eps.cuda()

    gru.initHidden(batch_size)
    # notice that 1st dim of gru outputs is seq_len, 2nd is batch_size
    z_M = gru(eps, 1).transpose(1, 0)
    z = torch.cat((z_M, z_C), 2)  # z.size() => (batch_size, n_frame, nz)
    return z.view(batch_size, 1, nz, 1, 1)


''' test models '''

for epoch in range(1, niter + 1):
    ''' prepare real images '''
    # real_videos.size() => (batch_size, nc, T, img_size, img_size)
    real_videos = random_choice()
    if cuda == True:
        real_videos = real_videos.cuda()
    real_videos = Variable(real_videos)

    # fake_videos.size() => (batch_size, nc, n_frames, img_size, img_size)
    fake_videos = real_videos[:, :, 0 : nPrevImg, :, :]
    prev_img = real_videos[:, :, 0 : nPrevImg, :, :]
    for i in range(nPrevImg, T):
        ''' Prepare fake images for every image'''
        # fake_image => Zi
        Zi = gen_zi()  # Z.size() => (batch_size, nz, 1, 1)
        Zi = Zi.contiguous().view(batch_size * 1, nz, 1, 1)
        # print ('The shape of Zi is:', Zi.shape)

        fake_img = gen_i(prev_img, Zi)
        # print ('The shape of fake_img is:', fake_img.shape)

        fake_img = fake_img.view(batch_size, -1, nc, 96, 96)
        fake_img = fake_img.transpose(2, 1)
        
        prev_img = torch.cat((prev_img[:, :, -nPrevImg+1:, :, :], fake_img), 2)

        fake_videos = torch.cat((fake_videos, fake_img), 2)

    print ('Complete the %d epoch', epoch)
    save_video(fake_videos[0].data.cpu().numpy().transpose(1, 2, 3, 0), epoch)
