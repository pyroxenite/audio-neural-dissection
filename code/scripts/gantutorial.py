# based on tutorial at https://www.tensorflow.org/tutorials/generative/dcgan?hl=fr

import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt 
import numpy as np

Tensor = torch.Tensor

device = "cuda" if torch.cuda.is_available() else "cpu"

BUFFER_SIZE = 30000
BATCH_SIZE  = 256
NUM_THREADS = 4

global_dataset = MNIST(root="./code/data", train=True, transform=ToTensor(), download=True)

valid_ratio: float = 0.2
nb_train = int((1.0 - valid_ratio) * len(global_dataset))
nb_valid = int(valid_ratio * len(global_dataset))

train_dataset, test_dataset = torch.utils.data.dataset.random_split(global_dataset, [nb_train, nb_valid])

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_THREADS)
test_loader  = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_THREADS)

class DebugLayer(nn.Module):
    def __init__(self, print_dims: bool = False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.print_dims = print_dims
        return
    
    def print_dimensions(self, x: Tensor) -> Tensor:
        print(f"Tensor dimensions : {x.size()}")

    def forward(self, x: Tensor) -> Tensor:
        
        if self.print_dims: self.print_dimensions(x)

        return x

class ReshapeLayer(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.shape = args    

    def forward(self, x: Tensor) -> Tensor:
        x = torch.reshape(x, self.shape)
        return x

class Generator(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(Generator, self).__init__(*args, **kwargs)

        self.input_shape = 100
        self.debug_print_dims: bool = True

        self.linear1 = nn.Linear(self.input_shape, 7*7*256, bias=False) 
        self.norm1 = nn.BatchNorm1d(7*7*256)
        self.act1 = nn.LeakyReLU()

        self.conv1 = nn.ConvTranspose2d(256, 128, kernel_size=5, stride=1, padding=2, bias=False)
        self.norm2 = nn.BatchNorm2d(128)
        self.act2 = nn.LeakyReLU()

        self.conv2 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False)
        self.norm3 = nn.BatchNorm2d(64) 
        self.act3 = nn.LeakyReLU() 

        self.conv3 = nn.ConvTranspose2d(64, 1, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False)
        self.act4 = nn.Tanh()

    def forward(self, x: Tensor) -> Tensor:

        x = self.linear1(x)
        x = self.norm1(x)
        x = self.act1(x)

        x = x.view(-1, 256, 7, 7)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.act2(x)

        x = self.conv2(x)
        x = self.norm3(x)
        x = self.act3(x)

        x = self.conv3(x)
        x = self.act4(x)

        return x

class Discriminator(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=2, bias=False)
        self.act1 = nn.LeakyReLU(0.2, inplace=True) 
        self.drop1 = nn.Dropout(0.3)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(128)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)
        self.drop2 = nn.Dropout(0.3)

        self.flat1 = nn.Flatten()
        self.linear1 = nn.Linear(2048, 128, bias=False)
        self.act3 = nn.LeakyReLU(0.2, inplace=True) 

        self.linear2 = nn.Linear(128, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        x = x.view(-1, 1, 28, 28)

        x = self.conv1(x)
        x = self.act1(x)
        x = self.drop1(x)

        x = self.conv2(x)
        x = self.batch_norm1(x)
        x = self.act2(x)
        x = self.drop2(x)
       
        x = self.flat1(x)
        x = self.linear1(x)
        x = self.act3(x)

        x = self.linear2(x)
        x = self.sigmoid(x)

        return x

def test_generator(model, n_samples) -> Tensor:
    noise = torch.randn((n_samples, model.input_shape), device=device)
    output = model(noise)
    return output

def train(generator, discriminator, n_epochs):
    
    loss_function = nn.BCELoss()

    generator = generator.to(device)
    discriminator = discriminator.to(device)

    for epoch in range(n_epochs):
        for n, (real_samples, _) in enumerate(train_loader):
            
            ### Training the discriminator
            discriminator_optimizer.zero_grad()
            real_samples = real_samples.to(device)
            curr_batch_size = real_samples.shape[0]
            noise = torch.randn((curr_batch_size, generator.input_shape), device=device)
            generated_images = generator(noise)
            disc_input = torch.cat((real_samples, generated_images))
            disc_labels = torch.cat((torch.ones((curr_batch_size, 1)), torch.zeros((curr_batch_size, 1)))).to(device)

            disc_output = discriminator(disc_input)
            disc_loss = loss_function(disc_output, disc_labels)
            disc_loss.backward(retain_graph=True)
            discriminator_optimizer.step()

            ### Training the generator
            generator_optimizer.zero_grad()
            noise = torch.randn((curr_batch_size, generator.input_shape), device=device)
            generated_images = generator(noise)
            
            disc_output = discriminator(generated_images)
            gen_loss = loss_function(disc_output, torch.ones((curr_batch_size, 1), device=device))
            gen_loss.backward()
            generator_optimizer.step()
            
            ### Training the generator again
            generator_optimizer.zero_grad()
            noise = torch.randn((curr_batch_size, generator.input_shape), device=device)
            generated_images = generator(noise)
            
            disc_output = discriminator(generated_images)
            gen_loss = loss_function(disc_output, torch.ones((curr_batch_size, 1), device=device))
            gen_loss.backward()
            generator_optimizer.step()

            if n%(len(train_loader)//5) == 0: print(f"> [Epoch {epoch+1}] Batch n°{n} ---- | Gen_loss : {gen_loss:.04f} || Disc_loss : {disc_loss:.04f} |")
        
        torch.save(generator, "code/models/backup/mnist-gan-generator.pt")
        torch.save(discriminator, "code/models/backup/mnist-gan-discriminator.pt")

    print("\n### Done Training ###\n")
    
    return generator, discriminator


if __name__ == "__main__":

    do_train = True
    use_pretrained = False # warning: is False, pretrained models will be replaced
    n_epochs = 10
    
    if do_train:
        
        print("\n### Start Training ###\n")

        if use_pretrained:
            generator = torch.load("code/models/mnist-gan-generator.pt")
            discriminator = torch.load("code/models/mnist-gan-discriminator.pt")
        else:
            generator = Generator()
            discriminator = Discriminator()

        generator_optimizer = torch.optim.Adam(generator.parameters(), 0.0005)
        discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), 0.00001)
        
        generator, discriminator = train(generator, discriminator, n_epochs)
        torch.save(generator, "code/models/mnist-gan-generator.pt")
        torch.save(discriminator, "code/models/mnist-gan-discriminator.pt")
    else:
        generator = torch.load("code/models/mnist-gan-generator.pt")
        discriminator = torch.load("code/models/mnist-gan-discriminator.pt")
    
    n_samples = 16
    result = test_generator(generator, n_samples)
    result = result.cpu()
    
    n_plot = int(np.sqrt(n_samples))
    fig, axes = plt.subplots(n_plot, n_plot)
    for i in range(n_plot):
        for j in range(n_plot):
            ax = axes[i, j]
            ax.imshow(result[0, 0, :, :].detach().numpy(), cmap='gray')
    plt.show()