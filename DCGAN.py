import torch
from torch import nn, optim
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
torch.manual_seed(0)

def show_tensor_images(image_tensor,num_images=25,size=(1,28,28)):
    image_tensor=(image_tensor+1)/2
    image_unflat=image_tensor.detach().cpu()
    image_grid=make_grid(image_unflat[:num_images],nrow=5)
    plt.imshow(image_grid.permute(1,2,0).squeeze())
    plt.show()

class Generator(nn.Module):
    def __init__(self, z_dim=10,im_chan=1,hidden_dim=128):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.generator=nn.Sequential(
            self.make_generator_block(z_dim,hidden_dim*4),
            self.make_generator_block(hidden_dim*4,hidden_dim*2,kernel_size=4,stride=1),
            self.make_generator_block(hidden_dim*2,hidden_dim),
            self.make_generator_block(hidden_dim,im_chan,kernel_size=4,last_layer=True),
        )
    def make_generator_block(self,input_dims,output_dims,kernel_size=3,stride=2,last_layer=False):
        if not last_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(input_dims,output_dims,kernel_size=kernel_size,stride=stride),
                nn.BatchNorm2d(output_dims),
                nn.ReLU(inplace=True),
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(input_dims,output_dims,kernel_size=kernel_size,stride=stride),
                nn.Tanh()
            )

    # def forward(self,x):
    #     x=x.view(len(x),self.z_dim,1,1)
    #     return self.generator(x)

    def unsqueeze_noise(self,noise):
        return noise.view(len(noise),self.z_dim,1,1)

    def forward(self,noise):
        x=self.unsqueeze_noise(noise)
        return self.generator(x)

class Discriminator(nn.Module):
    def __init__(self,im_chan=1,hidden_dim=32):
        super(Discriminator, self).__init__()
        self.discriminator=nn.Sequential(
            self.make_discriminator_block(im_chan,hidden_dim),
            self.make_discriminator_block(hidden_dim,hidden_dim*4),
            self.make_discriminator_block(hidden_dim*4,1,last_layer=True),
            #self.make_discriminator_block(hidden_dim*4,1,last_layer=True),
        )

    def make_discriminator_block(self,input_dims,output_dims,kernel_size=4,stride=2,last_layer=False):
        if not last_layer:
            return nn.Sequential(
                nn.Conv2d(input_dims,output_dims,kernel_size=kernel_size,stride=stride),
                nn.BatchNorm2d(output_dims),
                nn.LeakyReLU(0.2,inplace=True),
            )
        else:
            return nn.Sequential(
                nn.Conv2d(input_dims,output_dims,kernel_size=kernel_size,stride=stride),
            )

    def forward(self,x):
        pred=self.discriminator(x)
        return pred.view(len(pred),-1)

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m,nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight,0.0,0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight,0.0,0.02)
        torch.nn.init.constant_(m.bias,0)

def make_noise(batch_size,z_dim,device='mps'):
    return torch.randn(batch_size,z_dim,device=device)

def get_discriminator_loss(generator,discriminator,loss_function,real_image,batch_size,z_dim,device='mps'):
    noise=make_noise(batch_size,z_dim,device=device)
    fake=generator(noise)
    fake_disc=discriminator(fake.detach())
    real_disc=discriminator(real_image)
    real_loss=loss_function(real_disc,torch.ones_like(real_disc))
    fake_loss=loss_function(fake_disc,torch.zeros_like(fake_disc))
    total_loss=(real_loss+fake_loss)/2
    return total_loss

def get_generator_loss(generator,discriminator,loss_function,batch_size,z_dim,device='mps'):
    noise=make_noise(batch_size,z_dim,device=device)
    fake=generator(noise)
    fake_disc=discriminator(fake.detach())
    loss=loss_function(fake_disc,torch.ones_like(fake_disc))
    return loss

device='mps'
beta1=0.5
beta2=0.999
generator=Generator(z_dim=64).to(device)
discriminator=Discriminator().to(device)
generator_optimizer=optim.Adam(generator.parameters(),lr=0.0002,betas=(beta1,beta2))
discriminator_optimizer=optim.Adam(discriminator.parameters(),lr=0.0002,betas=(beta1,beta2))
generator=generator.apply(weights_init)
discriminator=discriminator.apply(weights_init)
loss_function=nn.BCEWithLogitsLoss()
transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])
dataloader=DataLoader(
    MNIST('./data',download=False,transform=transforms.ToTensor()),
    batch_size=128,
    shuffle=True
)
z_dim=64
n_epochs=1000
display_step=500
cur_step=0
mean_generator_loss=0
mean_discriminator_loss=0
for epoch in range(n_epochs+1):
    for image , _ in tqdm(dataloader):
        cur_batch_size=len(image)
        image=image.to(device)

        discriminator_optimizer.zero_grad()
        #disc_loss=get_discriminator_loss(generator,discriminator,loss_function,image,cur_batch_size,z_dim,device)
        noise=make_noise(cur_batch_size,z_dim,device)
        fake_noise=generator(noise)
        fake_disc=discriminator(fake_noise.detach())
        real_disc=discriminator(image)
        real_loss=loss_function(real_disc,torch.ones_like(real_disc))
        fake_loss=loss_function(fake_disc,torch.zeros_like(fake_disc))
        disc_loss=(real_loss+fake_loss)/2
        mean_discriminator_loss+=disc_loss.item()/display_step
        disc_loss.backward(retain_graph=True)
        discriminator_optimizer.step()

        generator_optimizer.zero_grad()
        #gen_loss=get_generator_loss(generator,discriminator,loss_function,cur_batch_size,z_dim,device)
        noise2=make_noise(cur_batch_size,z_dim,device)
        fake_noise2=generator(noise2)
        fake_disc2=discriminator(fake_noise2)
        gen_loss=loss_function(fake_disc2,torch.ones_like(fake_disc2))
        mean_generator_loss+=gen_loss.item()/display_step
        gen_loss.backward()
        generator_optimizer.step()

        if cur_step%display_step==0 and cur_step!=0:
            print(f"Display step: {cur_step}, Generator Loss: {mean_generator_loss}, Discriminator Loss: {mean_discriminator_loss}")
            mean_generator_loss=0
            mean_discriminator_loss=0
            # noise2=make_noise(cur_batch_size,z_dim,device)
            # fake_noise2=generator(noise2)
            show_tensor_images(image)
            show_tensor_images(fake_noise)
        cur_step+=1

# z_dim=64
# display_step=500
# n_epochs = 50
# cur_step = 0
# mean_generator_loss = 0
# mean_discriminator_loss = 0
# for epoch in range(n_epochs):
#     # Dataloader returns the batches
#     for real, _ in tqdm(dataloader):
#         cur_batch_size = len(real)
#         real = real.to(device)
#
#         ## Update discriminator ##
#         discriminator_optimizer.zero_grad()
#         fake_noise = make_noise(cur_batch_size, z_dim, device=device)
#         fake = generator(fake_noise)
#         disc_fake_pred = discriminator(fake.detach())
#         disc_fake_loss = loss_function(disc_fake_pred, torch.zeros_like(disc_fake_pred))
#         disc_real_pred = discriminator(real)
#         disc_real_loss = loss_function(disc_real_pred, torch.ones_like(disc_real_pred))
#         disc_loss = (disc_fake_loss + disc_real_loss) / 2
#
#         # Keep track of the average discriminator loss
#         mean_discriminator_loss += disc_loss / display_step
#         # Update gradients
#         disc_loss.backward(retain_graph=True)
#         # Update optimizer
#         discriminator_optimizer.step()
#
#         ## Update generator ##
#         generator_optimizer.zero_grad()
#         fake_noise_2 = make_noise(cur_batch_size, z_dim, device=device)
#         fake_2 = generator(fake_noise_2)
#         disc_fake_pred = discriminator(fake_2)
#         gen_loss = loss_function(disc_fake_pred, torch.ones_like(disc_fake_pred))
#         gen_loss.backward()
#         generator_optimizer.step()
#
#         # Keep track of the average generator loss
#         mean_generator_loss += gen_loss / display_step
#
#         ## Visualization code ##
#         if cur_step % display_step == 0 and cur_step > 0:
#             print(f"Step {cur_step}: Generator loss: {mean_generator_loss}, discriminator loss: {mean_discriminator_loss}")
#             show_tensor_images(fake)
#             show_tensor_images(real)
#             mean_generator_loss = 0
#             mean_discriminator_loss = 0
#         cur_step += 1