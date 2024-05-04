import os
import numpy as np
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch

IMAGES_PATH = 'images_WGAN_GP'
DATA_PATH = 'data/MNIST'
EPOCHS = 1000
BATCH_SIZE = 128
LEARNING_RATE = 0.0002
B1=0.5
B2=0.999
LATENT_DIM = 100
IMAGE_SIZE=28
CHANNELS=1
img_shape=(CHANNELS,IMAGE_SIZE,IMAGE_SIZE)
N_CRITIC=5
CLIP_VALUE=0.01
SAMPLE_INTERVAL=2500
LAMBDA_GP=10
DEVICE="mps"

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        def block(input,output,normalize=True):
            layers=[nn.Linear(input,output)]
            if normalize:
                layers.append(nn.BatchNorm1d(output,0.8))
            layers.append(nn.LeakyReLU(0.2,inplace=True))
            return layers
        self.model=nn.Sequential(
            *block(LATENT_DIM,128,normalize=False),
            *block(128,256),
            *block(256,512),
            *block(512,1024),
            nn.Linear(1024,int(np.prod(img_shape))),
            nn.Tanh()
        )
    def forward(self,noise):
        noise=self.model(noise)
        return noise.view(noise.size(0),*img_shape)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model=nn.Sequential(
            nn.Linear(int(np.prod(img_shape)),512),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(128, 1),
        )
    def forward(self,image):
        image=image.view(image.size(0),-1)
        image=self.model(image)
        return image

generator=Generator()
discriminator=Discriminator()

os.makedirs(DATA_PATH,exist_ok=True)
dataloader=torch.utils.data.DataLoader(
    datasets.MNIST(
        DATA_PATH,
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    ),
    batch_size=BATCH_SIZE,
    shuffle=True
)
os.makedirs(IMAGES_PATH,exist_ok=True)

optimizer_G=torch.optim.Adam(generator.parameters(),lr=LEARNING_RATE,betas=(B1,B2))
optimizer_D=torch.optim.Adam(discriminator.parameters(),lr=LEARNING_RATE,betas=(B1,B2))

Tensor=torch.FloatTensor

def compute_gradient_penalty(D,real_samples,fake_samples):
    alpha=Tensor(np.random.random((real_samples.size(0),1,1,1)))
    interpolates=(alpha*real_samples+((1-alpha)*fake_samples)).requires_grad_(True)
    d_interpolates=D(interpolates)
    fake=Variable(Tensor(real_samples.shape[0],1).fill_(1.0),requires_grad=False)
    gradients=autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients=gradients.view(gradients.size(0),-1)
    gradient_penalty=((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

img_list=[]
G_losses=[]
D_losses=[]
batches_done=0

for epoch in range(EPOCHS):
    for i, (images, labels) in enumerate(dataloader):
        z=Variable(Tensor(np.random.normal(0,1,(images.shape[0],LATENT_DIM))))
        valid=Variable(Tensor(images.shape[0],1).fill_(1.0),requires_grad=False)
        fake=Variable(Tensor(labels.shape[0],1).fill_(0.0),requires_grad=False)
        real_images=Variable(images.type(Tensor))
        fake_images=generator(z)

        optimizer_D.zero_grad()
        real_validity=discriminator(real_images)
        fake_validity=discriminator(fake_images)
        gradient_penalty=compute_gradient_penalty(discriminator,real_images.data,fake_images.data)
        loss_D=-torch.mean(real_validity)+torch.mean(fake_validity)+gradient_penalty*LAMBDA_GP
        loss_D.backward()
        optimizer_D.step()

        if i % N_CRITIC==0:
            optimizer_G.zero_grad()
            gen_images=generator(z)
            loss_G=-torch.mean(discriminator(gen_images))
            loss_G.backward()
            optimizer_G.step()

            print(
                "[Eopch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, EPOCHS, batches_done%len(dataloader), len(dataloader), loss_D.item(), loss_G.item())
            )

            G_losses.append(loss_G.item())
            D_losses.append(loss_D.item())

        if batches_done % SAMPLE_INTERVAL == 0:
            save_image(fake_images.data[:25], IMAGES_PATH+"/%07d.png"%batches_done, nrow=5, normalize=True)

            G_losses.append(loss_G.item())
            D_losses.append(loss_D.item())
            img_list.append(gen_images.data[:25])

        batches_done+=1


