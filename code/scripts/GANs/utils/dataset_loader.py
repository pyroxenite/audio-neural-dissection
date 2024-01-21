import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Compose, Normalize

def load_mnist():
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