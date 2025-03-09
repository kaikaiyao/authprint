import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, nz=100, ngf=128, channels_img=3):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # Input: N x nz x 1 x 1
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # State: N x (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # State: N x (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # State: N x (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # State: N x ngf x 32 x 32
            nn.ConvTranspose2d(ngf, channels_img, 4, 2, 1, bias=False),
            nn.Tanh()
            # Output: N x channels_img x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


def load_gan_model(self_trained_model_path, latent_dim):
    nz = latent_dim
    ngf = 128 # feature kernel size
    channels_img = 3

    netG = Generator(nz=nz, ngf=ngf, channels_img=channels_img)

    model_path = self_trained_model_path

    netG.load_state_dict(torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))

    return netG
