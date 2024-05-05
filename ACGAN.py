import os
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch

DATA_PATH="data/MNIST"
IMG_PATH="images_ACGAN"
Z_DIM=100
BATCH_SIZE=64
EPOCHS=200
LEARNING_RATE=0.0002
B1=0.5
B2=0.999
IMG_SIZE=32
CHANNELS=1
CLASS=10
IMG_SHAPE=(CHANNELS,IMG_SIZE,IMG_SIZE)
batches_done=0
SAMPLE_INTERVAL=500
DEVICE="mps"

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m,nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight,0.0,0.02)
    elif isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight,1.0,0.02)
        torch.nn.init.constant_(m.bias.data,0.0)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.label_emb=nn.Embedding(CLASS,Z_DIM)
        self.img_size=IMG_SIZE
        self.l1=nn.Sequential(nn.Linear(Z_DIM,128*self.img_size**2))
        def block(in_feat, out_feat,normalize=True):
            layers=[nn.ConvTranspose2d(in_feat, out_feat, kernel_size=3, stride=1, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat))
            layers.append(nn.LeakyReLU(0.2,inplace=True))
            return layers
        self.model=nn.Sequential(
            *block(128, 128, normalize=False),
            *block(128,64),
            *block(64,CHANNELS),
            # *block(32,CHANNELS),
            nn.Tanh()
        )
    def forward(self,noise,label):
        label_emb=self.label_emb(label)
        gen_input=torch.mul(label_emb,noise)
        out=self.l1(gen_input)
        out=out.view(out.shape[0],128,self.img_size,self.img_size)
        return self.model(out)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        def block(in_feat, out_feat, normalize=True):
            layers=[nn.Conv2d(in_feat, out_feat, kernel_size=3, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat))
            layers.append(nn.LeakyReLU(0.2,inplace=True))
            return layers
        self.size=IMG_SIZE//2**4
        self.adv_layer=nn.Sequential(
            nn.Linear(128*self.size**2, 1),
            nn.Sigmoid()
        )
        self.aux_layer=nn.Sequential(
            nn.Linear(128*self.size**2, CLASS),
            nn.Softmax()
        )
        self.model=nn.Sequential(
            *block(CHANNELS, 16, normalize=False),
            *block(16, 32),
            *block(32,64),
            *block(64,128),
        )
    def forward(self,img):
        img=self.model(img)
        img=img.view(img.shape[0],-1)
        validity=self.adv_layer(img)
        label=self.aux_layer(img)
        return validity,label

adversarial_loss=nn.BCELoss()
auxiliary_loss=nn.CrossEntropyLoss()
generator=Generator()
discriminator=Discriminator()
if torch.cuda.is_available():
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    auxiliary_loss.cuda()
# else:
#     generator.to(DEVICE)
#     discriminator.to(DEVICE)
#     adversarial_loss.to(DEVICE)
#     auxiliary_loss.to(DEVICE)
generator.apply(weights_init)
discriminator.apply(weights_init)
dataloader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(
        DATA_PATH,
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]),
    ),
    batch_size=BATCH_SIZE,
    shuffle=True
)
optimizer_G = torch.optim.Adam(generator.parameters(), lr=LEARNING_RATE,betas=(B1,B2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=LEARNING_RATE,betas=(B1,B2))
FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor

def sample_images(n_row,batches_done):
    z=Variable(FloatTensor(np.random.normal(0,1,(n_row**2,Z_DIM))))
    labels=np.array([num for _ in range(n_row) for num in range(n_row)])
    labels=Variable(LongTensor(labels))
    gen_imgs=generator(z,labels)
    save_image(gen_imgs.data,IMG_PATH+"%d.png"%batches_done,nrow=n_row,normalize=True)
    return gen_imgs

for epoch in range(EPOCHS):
    for i, (imgs, labels) in enumerate(dataloader):
        batch_size=imgs.shape[0]
        valid=Variable(FloatTensor(batch_size,1).fill_(1.0),requires_grad=False)
        fake=Variable(FloatTensor(batch_size,1).fill_(0.0),requires_grad=False)
        real_imgs=Variable(imgs.type(FloatTensor))
        real_labels=Variable(labels.type(LongTensor))
        optimizer_G.zero_grad()
        z=Variable(FloatTensor(np.random.normal(0,1,(batch_size,Z_DIM))))
        gen_labels=Variable(LongTensor(np.random.randint(0,CLASS,batch_size)))
        gen_imgs=generator(z,gen_labels)
        validity,pred_label=discriminator(gen_imgs)
        g_loss=0.5*(adversarial_loss(validity,valid)+auxiliary_loss(pred_label,gen_labels))
        g_loss.backward()
        optimizer_G.step()

        optimizer_D.zero_grad()
        real_pred,real_aux=discriminator(real_imgs)
        d_real_loss=(adversarial_loss(real_pred,valid)+auxiliary_loss(real_aux,labels))/2
        fake_pred,fake_aux=discriminator(gen_imgs.detach())
        d_fake_loss=(adversarial_loss(fake_pred,fake)+auxiliary_loss(fake_aux,gen_labels))/2
        d_loss=(d_real_loss+d_fake_loss)/2
        pred=np.concatenate([real_aux.data.cpu().numpy(),fake_aux.data.cpu().numpy()],axis=0)
        gt=np.concatenate([labels.data.cpu().numpy(),gen_labels.data.cpu().numpy()],axis=0)
        d_acc=np.mean(np.argmax(pred,axis=1)==gt)
        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %d%%] [G loss: %f]"
            %(epoch,EPOCHS,batches_done%len(dataloader),len(dataloader),d_loss.item(),d_acc*100,g_loss.item())
        )
        batches_done=epoch*len(dataloader)+i
        if batches_done%SAMPLE_INTERVAL==0:
            sample_images(n_row=10,batches_done=batches_done)


