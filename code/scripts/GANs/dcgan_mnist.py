import torch
import matplotlib.pyplot as plt 
import numpy as np

from utils.gan_struct import Generator, Discriminator, test_generator
from utils.dataset_loader import load_mnist
from utils.trainings import train_dcgan

device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":

    do_train = False
    use_pretrained = True # warning: if False, pretrained models will be replaced
    n_epochs = 200
    lr = 1e-4
    
    if do_train:

        print("\n### Start Training ###\n")

        train_loader, _ = load_mnist()
        
        if use_pretrained:
            generator = torch.load("code/models/DCGAN/mnist-gan-generator.pt")
            discriminator = torch.load("code/models/DCGAN/mnist-gan-discriminator.pt")
        else:
            generator = Generator()
            discriminator = Discriminator()

        generator_optimizer = torch.optim.Adam(generator.parameters(), lr=lr)
        discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr)
        
        generator, discriminator = train_dcgan(generator, generator_optimizer, discriminator, discriminator_optimizer, n_epochs, train_loader, device)
        torch.save(generator, "code/models/DCGAN/mnist-gan-generator.pt")
        torch.save(discriminator, "code/models/DCGAN/mnist-gan-discriminator.pt")
    else:
        generator = torch.load("code/models/DCGAN/mnist-gan-generator.pt")
        discriminator = torch.load("code/models/DCGAN/mnist-gan-discriminator.pt")
    
    n_samples = 16
    result = test_generator(generator, n_samples, device)
    result = result.cpu().detach()
    
    n_plot = int(np.sqrt(n_samples))
    fig, axes = plt.subplots(n_plot, n_plot)
    for i in range(n_plot):
        for j in range(n_plot):
            ax = axes[i, j]
            ax.imshow(result[i+4*j, 0, :, :]*127.5+127.5, cmap='gray')
    plt.show()