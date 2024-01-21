import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, input_shape=100, output_dim=1, hidden_dim=64) -> None:
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

def test_generator(model, n_samples, device):
    noise = torch.randn((n_samples, model.input_shape), device=device)
    output = model(noise)
    return output