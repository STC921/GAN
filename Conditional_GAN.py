import os
import numpy as np
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch

DATA_PATH="data/MNIST"
IMAGES_PATH="images_ConditionalGAN"
BATCH_SIZE=128
EPOCHS=2000
LEARNING_RATE=0.0002
B1=0.5
B2=0.999
Z_DIM=100
IMAGE_SIZE=28
CHANNELS=1
IMAGE_SHAPE=(CHANNELS,IMAGE_SIZE,IMAGE_SIZE)
SMAPLE_INTERVAL=20
CONDITION=10

cuda=True if torch.cuda.is_available() else False

class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()

        self.label_emb=nn.Embedding(CONDITION,CONDITION)

        def block(in_feat,out_feat,normalized=True):
            layer=[nn.Linear(in_feat,out_feat)]
            if normalized:
                layer.append(nn.BatchNorm1d(out_feat))
            layer.append(nn.LeakyReLU(0.2,inplace=True))
            return layer

        self.model=nn.Sequential(
            *block(Z_DIM+CONDITION,128,False),
            *block(128,256),
            *block(256,512),
            *block(512,1024),
            nn.Linear(1024,int(np.prod(IMAGE_SHAPE))),
            nn.Tanh()
        )
    def forward(self,noise,labels):
        get_input=torch.cat((self.label_emb(labels),noise),-1)
        input=self.model(get_input)
        input=input.view(input.size(0),*IMAGE_SHAPE)
        return input

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()

        self.label_emb = nn.Embedding(CONDITION,CONDITION)

        self.model=nn.Sequential(
            nn.Linear(CONDITION+int(np.prod(IMAGE_SHAPE)),512),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(512,256),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(256,128),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(128,1),
        )

    def forward(self,image,labels):
        d_in=torch.cat((image.view(image.size(0),-1),self.label_emb(labels)),-1)
        validity=self.model(d_in)
        return validity

loss_function=nn.MSELoss()
generator=Generator()
discriminator=Discriminator()
if cuda:
    generator.cuda()
    discriminator.cuda()
os.makedirs(DATA_PATH,exist_ok=True)
dataloader=torch.utils.data.DataLoader(
    datasets.MNIST(
        DATA_PATH,
        train=True,
        transform=transforms.Compose(
            [
                transforms.Resize(IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize([0.5],[0.5]),
            ]
        ),
    ),
    batch_size=BATCH_SIZE,
    shuffle=True,
)
os.makedirs(IMAGES_PATH,exist_ok=True)

optimizer_G=torch.optim.Adam(generator.parameters(),LEARNING_RATE,betas=(B1,B2))
optimizer_D=torch.optim.Adam(discriminator.parameters(),LEARNING_RATE,betas=(B1,B2))

FloatTensor=torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor=torch.cuda.LongTensor if cuda else torch.LongTensor

def sample_image(n_row,batches_done):
    z=Variable(FloatTensor(np.random.normal(0,1,(n_row**2,Z_DIM))))
    labels=np.array([num for _ in range(n_row) for num in range(n_row)])
    labels=Variable(LongTensor(labels))
    gen_imgs=generator(z,labels)
    save_image(gen_imgs.data,IMAGES_PATH+"/%07d.png"%batches_done,nrow=n_row,normalize=True)
    return gen_imgs.data

img_list=[]
G_losses=[]
D_losses=[]

for epoch in range(EPOCHS):
    for i, (image,labels) in enumerate(dataloader):
        batch_size=image.shape[0]
        z=Variable(FloatTensor(np.random.normal(0,1,(batch_size,Z_DIM))))
        gen_labels=Variable(LongTensor(np.random.randint(0,CONDITION,batch_size)))

        valid=Variable(FloatTensor(batch_size,1).fill_(1.0),requires_grad=False)
        fake=Variable(FloatTensor(batch_size,1).fill_(0.0),requires_grad=False)

        real_imgs=Variable(image.type(FloatTensor))
        labels=Variable(labels.type(LongTensor))

        optimizer_D.zero_grad()
        d_real_imgs=discriminator(real_imgs,labels)
        real_loss=loss_function(d_real_imgs,valid)
        fake_imgs=generator(z,gen_labels)
        d_fake_imgs=discriminator(fake_imgs.detach(),gen_labels)
        fake_loss=loss_function(d_fake_imgs,fake)
        d_loss=(real_loss+fake_loss)/2

        d_loss.backward()
        optimizer_D.step()

        optimizer_G.zero_grad()
        d_fake_imgs2=discriminator(fake_imgs,gen_labels)
        g_loss=loss_function(d_fake_imgs2,valid)

        g_loss.backward()
        optimizer_G.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [Generator loss %4f] [Discriminator loss %4f]"
            %(epoch,EPOCHS,i%len(dataloader),len(dataloader),g_loss.item(),d_loss.item())
        )

        batches_done=epoch*len(dataloader)+i
        if batches_done%SMAPLE_INTERVAL==0:
            img_data=sample_image(n_row=10,batches_done=batches_done)

            G_losses.append(g_loss.item())
            D_losses.append(d_loss.item())
            img_list.append(img_data)

            if batches_done>=600:
                SMAPLE_INTERVAL=1000

            if batches_done>=20000:
                SMAPLE_INTERVAL=2000

            if batches_done>=100000:
                SMAPLE_INTERVAL=5000






