# based on tutorial at https://www.tensorflow.org/tutorials/generative/dcgan?hl=fr

import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt 
import os
from tqdm import tqdm

Tensor = torch.Tensor

device = "cuda" if torch.cuda.is_available() else "cpu"

BUFFER_SIZE = 30000
BATCH_SIZE  = 16
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
        super().__init__(*args, **kwargs)

        self.input_shape = 100
        self.debug_print_dims: bool = True

        self.model = nn.Sequential(
            nn.Linear(self.input_shape, 7*7*256, bias=False), 
            DebugLayer(print_dims = self.debug_print_dims),
            nn.BatchNorm1d(7*7*256),
            nn.LeakyReLU(),

            nn.Unflatten(1, (7, 7, 256)), 
            DebugLayer(print_dims = self.debug_print_dims),

            nn.ConvTranspose2d(7, 128, 5, bias=False),
            DebugLayer(print_dims = self.debug_print_dims),
            nn.BatchNorm2d(128), 
            nn.LeakyReLU(), 

            nn.ConvTranspose2d(128, 64, 5, 2, bias=False),
            DebugLayer(print_dims = self.debug_print_dims),
            nn.BatchNorm2d(64), 
            nn.LeakyReLU(), 

            nn.ConvTranspose2d(64, 1, 5, 2, bias=False),
            DebugLayer(print_dims = self.debug_print_dims),
            nn.Tanh(),

            nn.Unflatten(1, torch.Size([28, 28, 1])),
            DebugLayer(print_dims = self.debug_print_dims),
        )

    def forward(self, x: Tensor) -> Tensor:
        output = self.model(x)  
        output = output.view(x.size(0), 1, 28, 28)
        return output 


class Discriminator(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.model = nn.Sequential(
            nn.Conv2d(1, 64, (5, 5), 2, bias=False),
            nn.LeakyReLU(), 
            nn.Dropout(0.3),

            nn.Conv2d(64, 1, (5, 5), 2, bias=False), 
            nn.LeakyReLU(), 
            nn.Dropout(0.3),

            nn.Flatten(), 
            nn.Linear(784, 10, bias=False), 
            nn.Softmax(dim=0)       
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x.view(-1, 28*28)
        return self.model(x)
    

generator = Generator()
discriminator = Discriminator()

loss_function = nn.CrossEntropyLoss()

def discriminator_loss(real_output: Tensor, fake_output: Tensor) -> Tensor:
    real_loss = loss_function(torch.ones_like(real_output), real_output)
    fake_loss = loss_function(torch.zeors_like(fake_output), fake_output)
    return real_loss + fake_loss

def generator_loss(fake_output: Tensor) -> Tensor:
    return loss_function(torch.ones_like(fake_output), fake_output)


generator_optimizer = torch.optim.Adam(generator.parameters(), 0.0001)
discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), 0.0001)


# Boucle d'entrainement 

do_train = True
nEpochs = 5
noise_dim = 100
num_examples_to_generate = 16

seed = torch.manual_seed(50)


def test_generator() -> Tensor:
    noise = torch.randn((BATCH_SIZE, generator.input_shape))
    output = generator(noise)
    return output


def test_discriminator() -> Tensor:
    input_image = torch.randn((28, 28))
    output = discriminator(input_image)
    return output

def train():
    
    print("Start Training\n")

    for epoch in tqdm(range(nEpochs)):
        if not do_train: return -1

        for n, (real_samples, real_labels) in enumerate(train_loader):
            
            noise = torch.randn((BATCH_SIZE, generator.input_shape,))

            generator_optimizer.zero_grad()
            discriminator_optimizer.zero_grad()

            generated_images = generator(noise)

            real_output = discriminator(real_samples)
            fake_output = discriminator(generated_images)

            gen_loss = generator_loss(fake_output)
            disc_loss = discriminator_loss(real_output, fake_output)


            gen_loss.backward()
            disc_loss.backward()

            generator_optimizer.step()
            discriminator_optimizer.step()

    print("\nDone Training\n")


if __name__ == "__main__":
    # train()
    result = test_generator()
    # result = test_discriminator()