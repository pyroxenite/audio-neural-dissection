# based on tutorial at https://www.tensorflow.org/tutorials/generative/dcgan

import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Compose, Normalize
import matplotlib.pyplot as plt 
import numpy as np

Tensor = torch.Tensor

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_mnist():
    BUFFER_SIZE = 30000
    BATCH_SIZE  = 256
    NUM_THREADS = 4

    # Normalizing images to [-1, 1] since the output of the generator is in [-1, 1] (tanh)
    transform = Compose([ToTensor(), Normalize((0.5,), (0.5,))])

    # Loading the MNIST dataset
    global_dataset = MNIST(root="./code/data", train=True, transform=transform, download=True)

    valid_ratio: float = 0.2
    nb_train = int((1.0 - valid_ratio) * len(global_dataset))
    nb_valid = int(valid_ratio * len(global_dataset))

    train_dataset, test_dataset = torch.utils.data.dataset.random_split(global_dataset, [nb_train, nb_valid])

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_THREADS)
    test_loader  = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_THREADS)
    
    return train_loader, test_loader

class Generator(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(Generator, self).__init__(*args, **kwargs)
        
        self.input_shape = 100
        
        self.lin_seq = nn.Sequential(
            nn.Linear(self.input_shape, 7*7*256, bias=False),
            nn.BatchNorm1d(7*7*256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.conv_seq = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(64, 1, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False),
            nn.Tanh()
        )
        
    def forward(self, x: Tensor) -> Tensor:
        x = self.lin_seq(x)
        x = x.view(-1, 256, 7, 7)
        x = self.conv_seq(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.main = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Conv2d(64, 128, kernel_size=5, stride=2, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Flatten(),
            nn.Linear(2048, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x.view(-1, 1, 28, 28)
        x = self.main(x)
        return x

def test_generator(model, n_samples) -> Tensor:
    noise = torch.randn((n_samples, model.input_shape), device=device)
    output = model(noise)
    return output

def train(generator, discriminator, n_epochs, train_loader):
    
    loss_function = nn.BCELoss()

    generator = generator.to(device)
    discriminator = discriminator.to(device)

    for epoch in range(n_epochs):
        n_print = len(train_loader)//5
        for n, (real_samples, _) in enumerate(train_loader):
            
            ### Training the discriminator
            # first on all-real samples
            real_samples = real_samples.to(device)
            curr_batch_size = real_samples.shape[0]
            
            discriminator.zero_grad()
            disc_output = discriminator(real_samples)
            disc_loss_real = loss_function(disc_output, torch.ones((curr_batch_size, 1), device=device))
            disc_loss_real.backward(retain_graph=True)
            disc_x = disc_output.mean().item()
            
            # then on all-fake samples
            noise = torch.randn((curr_batch_size, generator.input_shape), device=device)
            generated_samples = generator(noise)
            disc_output = discriminator(generated_samples)
            disc_loss_fake = loss_function(disc_output, torch.zeros((curr_batch_size, 1), device=device))
            disc_loss_fake.backward(retain_graph=True)
            d_g_z_1 = disc_output.mean().item()
            
            # disc_loss = disc_loss_real + disc_loss_fake
            discriminator_optimizer.step()

            ### Training the generator
            generator.zero_grad()
            disc_output = discriminator(generated_samples)
            gen_loss = loss_function(disc_output, torch.ones((curr_batch_size, 1), device=device))
            gen_loss.backward()
            generator_optimizer.step()
            d_g_z_2 = disc_output.mean().item()

            if n%n_print == 0:
                print(f"> [Epoch {epoch+1}] Batch n°{n} ---- | D(x) : {disc_x:.04f} || D(G(z))_1 : {d_g_z_1:.04f} || D(G(z))_2 : {d_g_z_2:.04f} |")
        
        torch.save(generator, "code/models/DCGAN/backup/mnist-gan-generator.pt")
        torch.save(discriminator, "code/models/DCGAN/backup/mnist-gan-discriminator.pt")

    print("\n### Done Training ###\n")
    
    return generator, discriminator

if __name__ == "__main__":

    do_train = False
    use_pretrained = True # warning: if False, pretrained models will be replaced
    n_epochs = 50
    
    if do_train:
        
        print("\n### Start Training ###\n")

        train_loader, _ = load_mnist()
        
        if use_pretrained:
            generator = torch.load("code/models/DCGAN/mnist-gan-generator.pt")
            discriminator = torch.load("code/models/DCGAN/mnist-gan-discriminator.pt")
        else:
            generator = Generator()
            discriminator = Discriminator()

        generator_optimizer = torch.optim.Adam(generator.parameters(), 1e-4)
        discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), 1e-4)
        
        generator, discriminator = train(generator, discriminator, n_epochs, train_loader)
        torch.save(generator, "code/models/DCGAN/mnist-gan-generator.pt")
        torch.save(discriminator, "code/models/DCGAN/mnist-gan-discriminator.pt")
    else:
        generator = torch.load("code/models/DCGAN/mnist-gan-generator.pt")
        discriminator = torch.load("code/models/DCGAN/mnist-gan-discriminator.pt")
    
    n_samples = 16
    result = test_generator(generator, n_samples)
    result = result.cpu().detach()
    
    n_plot = int(np.sqrt(n_samples))
    fig, axes = plt.subplots(n_plot, n_plot)
    for i in range(n_plot):
        for j in range(n_plot):
            ax = axes[i, j]
            ax.imshow(result[i+4*j, 0, :, :]*127.5+127.5, cmap='gray')
    plt.show()