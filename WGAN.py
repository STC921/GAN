import os
import numpy as np

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

IMAGES_PATH="images"
DATA_PATH="data/MNIST"

EPOCHS=1000
BATCH_SIZE=128
LEARNING_RATE=0.00005
LATENT_DIM=100
IMAGE_SIZE=28
CHANNELS=1
image_shape=(CHANNELS,IMAGE_SIZE,IMAGE_SIZE)

N_CRITIC=5
CLIP_VALUE=0.01
SAMPLE_INTERNAL=2500
DEVICE="mps"

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat,0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        self.model = nn.Sequential(
            *block(LATENT_DIM, 128, normalize=False),
            *block(128,256),
            *block(256,512),
            *block(512,1024),
            nn.Linear(1024,np.prod(image_shape)),
            nn.Tanh()
        )
    def forward(self, input):
        x=self.model(input)
        return x.view(x.size(0),*image_shape)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(np.prod(image_shape),512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
        )
    def forward(self, image):
        image=image.view(image.size(0),-1)
        return self.model(image)

generator=Generator().to(DEVICE)
discriminator=Discriminator().to(DEVICE)

os.makedirs(DATA_PATH,exist_ok=True)
dataloader=torch.utils.data.DataLoader(
    datasets.MNIST(
        DATA_PATH,
        train=True,
        transform=transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ]),
        download=True
    ),
    batch_size=BATCH_SIZE,
    shuffle=True,
)
os.makedirs(IMAGES_PATH,exist_ok=True)

optimizer_G=torch.optim.RMSprop(generator.parameters(), lr=LEARNING_RATE)
optimizer_D=torch.optim.RMSprop(discriminator.parameters(), lr=LEARNING_RATE)

Tensor=torch.FloatTensor

img_list=[]
G_losses=[]
D_losses=[]

batches_done=0

for epoch in range(EPOCHS):
    for i, (images, labels) in enumerate(dataloader):
        z=Variable(Tensor(np.random.normal(0, 1, (images.shape[0], LATENT_DIM)))).to(DEVICE)
        vaild=Variable(Tensor(images.shape[0],1).fill_(1.0),requires_grad=False).to(DEVICE)
        fake=Variable(Tensor(images.shape[0],1).fill_(0.0),requires_grad=False).to(DEVICE)

        real_images=Variable(images.type(Tensor)).to(DEVICE)
        fake_images=generator(z)
        optimizer_D.zero_grad()
        loss_D=-torch.mean(discriminator(real_images))+torch.mean(discriminator(fake_images))
        loss_D.backward()
        optimizer_D.step()

        for p in discriminator.parameters():
            p.data.clamp_(-CLIP_VALUE,CLIP_VALUE)
        if i%N_CRITIC==0:
            optimizer_G.zero_grad()
            gen_images=generator(z)
            loss_G=-torch.mean(discriminator(gen_images))
            loss_G.backward()
            optimizer_G.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, EPOCHS, batches_done%len(dataloader), len(dataloader), loss_D.item(), loss_G.item())
            )

            G_losses.append(loss_G.item())
            D_losses.append(loss_D.item())

        if batches_done%SAMPLE_INTERNAL==0:
            save_image(fake_images.data[:25],IMAGES_PATH+"/%07d.png"%batches_done,nrow=5,normalize=True)
            G_losses.append(loss_G.item())
            D_losses.append(loss_D.item())
            img_list.append(fake_images.data[:25])

        batches_done+=1

