import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Compose, Normalize
import matplotlib.pyplot as plt 
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

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

class Generator(nn.Module):
    def __init__(self, input_shape=100, output_dim=1, hidden_dim=64):
        super(Generator, self).__init__()
        
        self.input_shape = input_shape
        self.hidden_dim = hidden_dim
        
        self.lin_seq = nn.Sequential(
            nn.Linear(self.input_shape, 7*7*4*hidden_dim, bias=False),
            nn.BatchNorm1d(7*7*4*hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.conv_seq = nn.Sequential(            
            nn.ConvTranspose2d(4*hidden_dim, 2*hidden_dim, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(2*hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(2*hidden_dim, hidden_dim, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(hidden_dim, output_dim, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False),
            nn.Tanh(),
        )
        
    def forward(self, x):
        x = self.lin_seq(x)
        x = x.view(-1, 4*self.hidden_dim, 7, 7)
        x = self.conv_seq(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=16):
        super(Discriminator, self).__init__()
        
        self.main = nn.Sequential(
            nn.Conv2d(input_dim, 4*hidden_dim, kernel_size=5, stride=2, bias=False),
            nn.BatchNorm2d(4*hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Conv2d(4*hidden_dim, 8*hidden_dim, kernel_size=5, stride=2, bias=False),
            nn.BatchNorm2d(8*hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Flatten(),
            nn.Linear(8*hidden_dim*hidden_dim, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = self.main(x)
        return x

def gradient_penalty(gradient):
    gradient = gradient.view(len(gradient), -1)
    gradient_norm = gradient.norm(2, dim=1)
    penalty = torch.mean((gradient_norm-1)**2)
    return penalty

def test_generator(model, n_samples):
    noise = torch.randn((n_samples, model.input_shape), device=device)
    output = model(noise)
    return output

def get_gen_loss(fake_pred):
    gen_loss = -1*torch.mean(fake_pred)
    return gen_loss

def get_disc_loss(fake_pred, real_pred, penalty, c_lambda):
    disc_loss = torch.mean(fake_pred) - torch.mean(real_pred) + c_lambda*penalty
    return disc_loss

def get_gradient(discriminator, real_samples, fake_samples, eps):
    mixed_samples = real_samples*eps + fake_samples*(1-eps)
    mixed_scores = discriminator(mixed_samples)
    gradient = torch.autograd.grad(
        inputs=mixed_samples,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    return gradient

def train(generator, discriminator, n_epochs, z_dim):
    
    loss_function = nn.BCELoss()
    c_lambda = 10
    n_train_disc = 5

    generator = generator.to(device)
    discriminator = discriminator.to(device)

    for epoch in range(n_epochs):
        n_print = len(train_loader)//5
        for n, (real_samples, _) in enumerate(train_loader):
            
            real_samples = real_samples.to(device)
            curr_batch_size = len(real_samples)
            
            ### Training the discriminator
            mean_disc_loss = 0
            for _ in range(n_train_disc):
                disc_optimizer.zero_grad()
                
                disc_real_pred = discriminator(real_samples)
                noise = torch.randn(curr_batch_size, z_dim, device=device)
                fake_images = generator(noise)
                disc_fake_pred = discriminator(fake_images)
                
                eps = torch.rand(curr_batch_size, 1, 1, 1, device=device, requires_grad=True)
                gradient = get_gradient(discriminator, real_samples, fake_images.detach(), eps)
                grad_penalty = gradient_penalty(gradient)
                
                disc_loss = get_disc_loss(disc_fake_pred, disc_real_pred, grad_penalty, c_lambda)
                
                disc_loss.backward(retain_graph=True)
                disc_optimizer.step()
                
                mean_disc_loss += disc_loss.item() / n_train_disc
                
            ### Training the generator
            gen_optimizer.zero_grad()
            noise = torch.randn(curr_batch_size, z_dim, device=device)
            fake_images = generator(noise)
            disc_fake_pred = discriminator(fake_images)
            
            gen_loss = get_gen_loss(disc_fake_pred)
            
            gen_loss.backward()
            gen_optimizer.step()

            if n%n_print == 0:
                print(f"> [Epoch {epoch+1}] Batch n°{n} ---- | Disc_loss : {mean_disc_loss:.04f} || Gen_loss : {gen_loss.item():.04f} |")
        
        torch.save(generator, "code/models/WGAN/backup/mnist-wgan-generator.pt")
        torch.save(discriminator, "code/models/WGAN/backup/mnist-wgan-discriminator.pt")

    print("\n### Done Training ###\n")
    
    return generator, discriminator


if __name__ == "__main__":

    do_train = True
    use_pretrained = True # warning: if False, pretrained models will be replaced
    n_epochs = 50
    lr = 1e-4
    
    z_dim = 100
    
    if do_train:
        print("\n### Start Training ###\n")

        if use_pretrained:
            generator = torch.load("code/models/WGAN/mnist-wgan-generator.pt")
            discriminator = torch.load("code/models/WGAN/mnist-wgan-discriminator.pt")
        else:
            generator = Generator(z_dim)
            discriminator = Discriminator()

        gen_optimizer = torch.optim.Adam(generator.parameters(), lr=lr)
        disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr)
        
        generator, discriminator = train(generator, discriminator, n_epochs, z_dim)
        torch.save(generator, "code/models/WGAN/mnist-wgan-generator.pt")
        torch.save(discriminator, "code/models/WGAN/mnist-wgan-discriminator.pt")
    else:
        generator = torch.load("code/models/WGAN/mnist-wgan-generator.pt")
        discriminator = torch.load("code/models/WGAN/mnist-wgan-discriminator.pt")
    
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