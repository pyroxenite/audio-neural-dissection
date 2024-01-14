# based on tutorial at https://www.tensorflow.org/tutorials/generative/dcgan?hl=fr

import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt 
import os
from os.path import isdir, isfile
from tqdm import tqdm
import numpy as np 
from numpy.typing import NDArray

Tensor = torch.Tensor


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

        x = torch.reshape(x, (BATCH_SIZE, 256, 7, 7))
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
        self.act1 = nn.LeakyReLU() 
        self.drop1 = nn.Dropout(0.3)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, bias=False) 
        self.act2 = nn.LeakyReLU() 
        self.drop2 = nn.Dropout(0.3)

        self.flat1 = nn.Flatten()
        self.linear1 = nn.Linear(2048, 1, bias=False)
        # self.act3 = nn.LeakyReLU()

        # self.linear2 = nn.Linear(128, 1, bias=False)
        self.softmax = nn.Softmax(dim=0)       


    def forward(self, x: Tensor) -> Tensor:
        x = x.view(BATCH_SIZE, 1, 28, 28)

        x = self.conv1(x)
        x = self.act1(x)
        x = self.drop1(x)

        x = self.conv2(x)
        x = self.act2(x)
        x = self.drop2(x)
       
        x = self.flat1(x)
        x = self.linear1(x)
        # x = self.act3(x)

        # x = self.linear2(x)
        x = self.softmax(x)

        return x


def discriminator_loss(real_output: Tensor, fake_output: Tensor, loss_function) -> Tensor:
    real_loss = loss_function(torch.ones_like(real_output), real_output)
    fake_loss = loss_function(torch.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss


def generator_loss(fake_output: Tensor, loss_function) -> Tensor:
    return loss_function(torch.ones_like(fake_output), fake_output)


def test_generator(model) -> Tensor:
    noise = torch.randn((BATCH_SIZE, model.input_shape))
    output = model(noise)
    return output


def test_discriminator(model) -> Tensor:
    input_image = torch.randn((BATCH_SIZE, 28, 28))
    output = model(input_image)
    return output


def train(generator: Generator, discriminator: Discriminator, n_epochs: int) -> tuple[Tensor, Tensor]:
    
    loss_function = nn.CrossEntropyLoss()

    gen_loss_array: Tensor = torch.zeros(n_epochs)
    disc_loss_array: Tensor = torch.zeros(n_epochs)

    generator = generator.to(device)
    discriminator = discriminator.to(device)


    for epoch in range(n_epochs):
        if not do_train: break

        print(f"epoch : {epoch}/{n_epochs}")

        gen_epoch_loss = 0.0
        disc_epoch_loss = 0.0

        for n, (real_samples, real_labels) in enumerate(train_loader):
            
            noise = torch.randn((BATCH_SIZE, generator.input_shape,)).to(device)
            real_samples = real_samples.to(device)

            generator_optimizer.zero_grad()
            discriminator_optimizer.zero_grad()

            generated_images = generator(noise)

            real_output = discriminator(real_samples)
            fake_output = discriminator(generated_images)

            gen_loss = generator_loss(torch.flatten(fake_output), loss_function)
            disc_loss = discriminator_loss(torch.flatten(real_output), torch.flatten(fake_output), loss_function)

            gen_loss.backward(retain_graph=True)
            disc_loss.backward()

            generator_optimizer.step()
            discriminator_optimizer.step()


            gen_epoch_loss += torch.sum(gen_loss)
            disc_epoch_loss += torch.sum(disc_loss)
        
        print(f"gen_epoch_loss = {gen_epoch_loss}")
        print(f"disc_epoch_loss = {disc_epoch_loss}")
        gen_loss_array[epoch] = gen_epoch_loss
        disc_loss_array[epoch] = disc_epoch_loss

        if epoch % 10 == 0 and save_models:

            torch.save(generator.state_dict(), generator_file)
            torch.save(discriminator.state_dict(), discriminator_file)


    return (gen_loss_array, disc_loss_array)


if __name__ == "__main__":

    do_train = True
    load_models = False
    save_models = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Computation device : {device}\n")


    BUFFER_SIZE = 30000
    BATCH_SIZE  = 16
    NUM_THREADS = 4

    global_dataset = MNIST(root="./code/data", train=True, transform=ToTensor(), download=True)

    valid_ratio: float = 0.2
    nb_train = int((1.0 - valid_ratio) * len(global_dataset))
    nb_valid = int(valid_ratio * len(global_dataset))


    train_dataset, test_dataset = torch.utils.data.dataset.random_split(global_dataset, [nb_train, nb_valid])

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_THREADS)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_THREADS)


    generator_file = "code/models/mnist_gan_generator_benjamin.pt"
    discriminator_file = "code/models/mnist_gan_discriminator_benjamin.pt"

    n_epochs = 20
    print(f"n_epochs = {n_epochs}")

    generator = None
    discriminator = None

    if (load_models and isfile(generator_file) 
                    and isfile(discriminator_file)):

        generator = torch.load(generator_file)
        discriminator = torch.load(discriminator_file)

    else:
        generator = Generator()
        discriminator = Discriminator()


    generator_optimizer = torch.optim.Adam(generator.parameters(), 0.001)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), 0.001)

    seed = torch.manual_seed(50)

    print("Start Training\n")
    gen_loss, disc_loss = train(generator, discriminator, n_epochs)
    # result = test_generator(generator)
    # result = test_discriminator(discriminator)
    print("\nDone Training\n")

    torch.save(generator.state_dict(), generator_file)
    torch.save(discriminator.state_dict(), discriminator_file)

    fig, ax = plt.subplots()
    ax.plot(gen_loss.detach().numpy())
    ax.plot(disc_loss.detach().numpy())

    plt.show(block=True)