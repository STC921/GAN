import torch
from torch import nn
from torch import optim
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
torch.manual_seed(0) #set for testing purposes

torch.manual_seed(0)

def show_tensor_images(image_tensor,num_images=25,size=(1,28,28)):
    image_unflat=image_tensor.detach().cpu().view(-1,*size)
    image_grid=make_grid(image_unflat[:num_images],nrow=5)
    plt.imshow(image_grid.permute(1,2,0).squeeze())
    plt.show()

class Generator(nn.Module):
    def __init__(self,z_dim=10,image_shape=784,hidden_dim=128):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.channels = image_shape
        self.generate=nn.Sequential(
            self.make_generator_block(z_dim,hidden_dim),
            self.make_generator_block(hidden_dim,hidden_dim*2),
            self.make_generator_block(hidden_dim*2,hidden_dim*4),
            self.make_generator_block(hidden_dim*4,hidden_dim*8),
            self.make_generator_block(hidden_dim*8,image_shape,True),
        )

    def make_generator_block(self,input_dim,output_dim,last_layer=False):
        if not last_layer:
            return nn.Sequential(
                nn.Linear(input_dim,output_dim),
                nn.BatchNorm1d(output_dim),
                nn.LeakyReLU(0.2,inplace=True),
            )
        else:
            return nn.Sequential(
                nn.Linear(input_dim,output_dim),
                nn.Sigmoid()
            )

    def forward(self,x):
        x=self.generate(x)
        return x

    def get_gen(self):
        return self.generate

class Discriminator(nn.Module):
    def __init__(self,channels=784,hidden_dim=128):
        super(Discriminator, self).__init__()
        self.discriminate=nn.Sequential(
            self.make_discriminator_block(channels,hidden_dim*4),
            self.make_discriminator_block(hidden_dim*4,hidden_dim*2),
            self.make_discriminator_block(hidden_dim*2,hidden_dim),
            self.make_discriminator_block(hidden_dim,1,last_layer=True),
        )

    def make_discriminator_block(self,input_dim,output_dim,last_layer=False):
        if not last_layer:
            return nn.Sequential(
                nn.Linear(input_dim,output_dim),
                nn.LeakyReLU(0.2,inplace=True),
            )
        else:
            return nn.Sequential(
                nn.Linear(input_dim,output_dim),
                nn.Sigmoid()
            )

    def forward(self,x):
        x=self.discriminate(x)
        return x
    def get_disc(self):
        return self.discriminate

def make_noise(nums,dims,device='cpu'):
    return torch.randn(nums,dims,device=device)

# loss=torch.nn.BCELoss()
loss=nn.BCEWithLogitsLoss()
device='cpu'
batch_size=128
z_dims=64
generator=Generator(z_dims).to(device)
discriminator=Discriminator().to(device)
generator_optimizer=optim.Adam(generator.parameters(),lr=0.0002,betas=(0.5,0.999))
discriminator_optimizer=optim.Adam(discriminator.parameters(),lr=0.0002,betas=(0.5,0.999))
dataloader=DataLoader(
    MNIST('./data',download=False,transform=transforms.ToTensor()),
    batch_size=batch_size,
    shuffle=True
)
# G_loss=[]
# D_loss=[]
# mean_gen_loss=0
# mean_disc_loss=0
# for epoch in range(0,201):
#     for images,_ in tqdm(dataloader):
#         cur_batch_size=len(images)
#         real=images.view(cur_batch_size,-1).to(device)
#         fake_noise=make_noise(cur_batch_size,z_dims,device)
#         fake=generator(fake_noise)
#         fake_pred=discriminator(fake)
#         print(discriminator(real).shape,discriminator(fake).shape,torch.ones_like(real).shape)
#         discriminator_optimizer.zero_grad()
#         real_loss=loss1(discriminator(real).to(device),torch.ones(real.shape).to(device))
#         fake_loss=loss1(discriminator(fake.detach()),torch.zeros_like(fake))
#         disc_loss=real_loss+fake_loss/2
#         disc_loss.backward()
#         discriminator_optimizer.step()
#         generator_optimizer.zero_grad()
#         noise_2=make_noise(cur_batch_size,z_dims,device)
#         fake_noise_2=make_noise(cur_batch_size,z_dims,device)
#         fake_2=generator(fake_noise_2)
#         gen_loss=loss1(discriminator(fake_2),torch.zeros_like(fake_2))
#         gen_loss.backward()
#         generator_optimizer.step()
#         mean_gen_loss+=gen_loss.item()/500
#         mean_disc_loss+=disc_loss.item()/500
#         if epoch%500==0 and epoch!=0:
#             print(f"Turns: {epoch} Generator loss: {mean_gen_loss} Discriminator loss: {mean_disc_loss}")
#             G_loss.append(mean_gen_loss)
#             D_loss.append(mean_disc_loss)
#             mean_gen_loss=0
#             mean_disc_loss=0
#             fake_noise=make_noise(cur_batch_size,z_dims,device)
#             fake=generator(fake_noise)
#             show_tensor_images(fake)
#             show_tensor_images(real)
#     epoch+=1


# def get_disc_loss(gen,disc,criterion,real,num_images,z_dim,device='mps'):
#     fake_noise=make_noise(num_images,z_dim,device)
#     fake=gen(fake_noise)
#     disc_fake_pred=disc(fake.detach())
#     disc_real_pred=disc(real).to(device)
#     disc_fake_loss=criterion(disc_fake_pred,torch.zeros(disc_fake_pred.shape).to(device))
#     disc_real_loss=criterion(disc_real_pred,torch.ones(disc_real_pred.shape).to(device))
#     disc_loss=(disc_fake_loss+disc_real_loss)/2
#     return disc_loss
#
# def get_gen_loss(gen,disc,criterion,num_images,z_dim,device='mps'):
#     fake_noise=make_noise(num_images,z_dim,device)
#     fake=gen(fake_noise)
#     disc_fake_pred=disc(fake)
#     gen_loss=criterion(disc_fake_pred,torch.ones(disc_fake_pred.shape).to(device))
#     return gen_loss
#
# display_step=500
# n_epochs=200
# cur_step=0
# mean_generator_loss=0
# mean_discriminator_loss=0
# test_generator=True
# gen_loss=False
# error=False
# for epoch in range(n_epochs):
#     for real, _ in tqdm(dataloader):
#         cur_batch_size=len(real)
#         real=real.view(cur_batch_size,-1).to('mps')
#         discriminator_optimizer.zero_grad() #清空梯度缓存
#         print(discriminator(real).shape,real.shape)
#         disc_loss=get_disc_loss(generator,discriminator,loss1,real,cur_batch_size,z_dims,device) #计算误差
#         disc_loss.backward(retain_graph=True) #计算梯度
#         discriminator_optimizer.step() #使用优化器更新模型参数
#
#         if test_generator:
#             old_generator_weights=generator.generate[0][0].weight.data.clone()
#
#         generator_optimizer.zero_grad() #清空梯度缓存
#         gen_loss=get_gen_loss(generator,discriminator,loss1,cur_batch_size,z_dims,device)
#         gen_loss.backward() #计算梯度
#         generator_optimizer.step() #使用优化器更新模型参数
#
#         # if test_generator:
#         #     try:
#         #         assert lr>0.0000002 or (gen.gen[0][0].weight.grad.abs().max()<0.0005 and epoch==0)
#         #         assert torch.any(gen.gen[0][0].weight.detach().clone()!=old_generator_weights)
#         #     except:
#         #         error=True
#         #         print("runtime tests have failed")
#
#         mean_discriminator_loss+=disc_loss.item()/display_step
#         mean_generator_loss+=gen_loss.item()/display_step
#
#         if cur_step%display_step==0 and cur_step >0:
#             print(f"step {cur_step}: Generator loss: {mean_generator_loss},discriminator loss: {mean_discriminator_loss}")
#             fake_noise=make_noise(cur_batch_size,z_dims,device)
#             fake=generator(fake_noise)
#             show_tensor_images(fake)
#             show_tensor_images(real)
#             mean_generator_loss=0 #初始化生成器误差均值
#             mean_discriminator_loss=0 #初始化判别器误差均值
#         cur_step+=1

def get_discriminator_loss(generator,discriminator,loss_function,real,batch_size,z_dims,device='mps'):
    noise=make_noise(batch_size,z_dims,device=device)
    fake_noise=generator(noise)
    fake_disc=discriminator(fake_noise.detach())
    real_disc=discriminator(real)
    fake_loss=loss_function(fake_disc,torch.zeros_like(fake_disc))
    real_loss=loss_function(real_disc,torch.ones_like(real_disc))
    loss=(fake_loss+real_loss)/2
    return loss

def get_generator_loss(generator,discriminator,loss_function,batch_size,z_dims,device='mps'):
    noise=make_noise(batch_size,z_dims,device=device)
    fake_noise=generator(noise)
    fake_disc=discriminator(fake_noise)
    loss=loss_function(fake_disc,torch.ones_like(fake_disc))
    return loss

e_epochs=1000
display_step=500
mean_gen_loss=0
mean_disc_loss=0
cur_step=0

for epoch in range(e_epochs+1):
    for image,_ in tqdm(dataloader):
        cur_batch_size=len(image)
        image=image.view(cur_batch_size,-1)

        discriminator_optimizer.zero_grad()
        disc_loss=get_discriminator_loss(generator,discriminator,loss,image,batch_size,z_dims,device)
        mean_disc_loss+=disc_loss/display_step
        disc_loss.backward()
        discriminator_optimizer.step()

        generator_optimizer.zero_grad()
        gen_loss=get_generator_loss(generator,discriminator,loss,batch_size,z_dims,device)
        mean_gen_loss+=gen_loss/display_step
        gen_loss.backward()
        generator_optimizer.step()

        if cur_step % display_step ==0 and cur_step!=0:
            noise2=make_noise(cur_batch_size,z_dims,device=device)
            fake2=generator(noise2)
            show_tensor_images(fake2)
            show_tensor_images(image)
            print(f"Epoch: {epoch}, Generator loss: {gen_loss}, Discriminator loss: {disc_loss}")
            mean_gen_loss=0
            mean_disc_loss=0
        cur_step+=1














