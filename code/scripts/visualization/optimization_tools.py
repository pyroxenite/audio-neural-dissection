import torch
from tqdm import tqdm
import numpy as np
import os
import matplotlib.pyplot as plt
from IPython.display import clear_output
from torchvision.transforms import v2

LAYER_OBJ = 1
CHANNEL_OBJ = 2
NEURON_OBJ = 3

class Objective():
    def __init__(self, obj_type, layer, **kwargs):
        self.type = obj_type
        self.layer = layer
        if obj_type == LAYER_OBJ:
            pass
        elif obj_type == CHANNEL_OBJ:
            self.channel_index = kwargs["channel_index"]
        elif obj_type == NEURON_OBJ:
            self.neuron_index = kwargs["neuron_index"]
        else:
            raise ValueError("`obj_type` must be one of 1 (LAYER_OBJ), 2 (CHANNEL_OBJ), 3 (NEURON_OBJ).")


########## GANs ##########

class GANActivationMaximizer():
    def __init__(self, model, input_shape):
        super().__init__()

        self.model = model
        self.input_shape = input_shape

    def optimize(self, objective, transforms=[], iterations=10, eta=1e-4, device="cpu"):
        start = torch.randn(self.input_shape)*0.01
        self.input = torch.nn.Parameter(start.clone(), requires_grad=True)

        def hook(model, input, output):
            global layer_out
            layer_out = output
        
        objective.layer.register_forward_hook(hook)

        optimizer = torch.optim.Adam([self.input], eta)

        for _ in tqdm(range(iterations)):
            for t in transforms:
                self.input.data = t(self.input.data)

            self.model.zero_grad()
            self.model.forward(self.input.to(device))

            if objective.type == LAYER_OBJ:        
                loss = -torch.mean(layer_out**2)
            elif objective.type == CHANNEL_OBJ:
                mask = torch.zeros_like(layer_out)
                mask[0, objective.channel_index, :, :] = 1
                loss = -torch.mean((mask * layer_out)**2)
            elif objective.type == NEURON_OBJ:
                mask = torch.zeros_like(layer_out)
                mask[objective.neuron_index] = 1
                loss = -torch.mean((mask * layer_out)**2)

            loss.backward(retain_graph=True)
            optimizer.step()

        return start, self.input


########## GoogleNet ##########

def _random_translation(offset, max_i, max_j, std=5):
    return [
        np.minimum(max_i, np.maximum(0, offset[0] + np.random.randn() * std)),
        np.minimum(max_j, np.maximum(0, offset[1] + np.random.randn() * std)),
    ]

def _normalize(im):
    # im = im - torch.mean(im)
    # im = im / torch.std(im)/10
    # im += 0.5
    im -= torch.min(im)
    im /= torch.max(im) - torch.min(im)
    return im

class GoogleNetActivationMaximizer():
    def __init__(self, model, input_shape):
        super().__init__()

        self.model = model
        self.input_shape = input_shape

    def plot_input(self, i, plot_period, save=False, plot_inline=False):
        im = np.array(self.input.data)[0]
        
        if plot_inline or save:
            plt.imshow(np.moveaxis(im, 0, -1))
            plt.title(f"Iteration: {i+1}")
            
            if save:
                if not os.path.isdir("output"):
                    os.mkdir("output")
                plt.savefig(f"output/im{i//plot_period:04d}.png")
            if plot_inline:
                clear_output()
                plt.show()

    def optimize(self, objective, transforms=[], iterations=10, eta=1e-8, loss_func=None, reg=0, plot_period=None, save=False, plot_inline=False, start_image=None, device="cpu"):
        larger_shape = list(self.input_shape)
        larger_shape[-1] = int(1.2 * larger_shape[-1])
        larger_shape[-2] = int(1.2 * larger_shape[-2])

        offset = [(larger_shape[-2] - self.input_shape[-2])//2, (larger_shape[-1] - self.input_shape[-1])//2]

        if start_image != None:
            start = start_image.reshape(larger_shape)
        else:
            start = torch.randn(larger_shape)*0.01 + 0.5

        self.input = torch.nn.Parameter(start.clone(), requires_grad=True)

        if start_image == None:
            for i in range(3, 40, 2):
                self.input.data = v2.GaussianBlur(i, sigma=(i/2, i/2))(self.input.data)

        loss_history = np.zeros(iterations)

        def hook(model, input, output):
            global layer_out
            layer_out = output
        
        objective.layer.register_forward_hook(hook)

        optimizer = torch.optim.Adam([self.input], eta)

        if loss_func == None:
            loss_func = lambda x: -torch.mean(x**2)
        
        for i in tqdm(range(iterations)):
            try:
                # Normalization
                self.input.data = _normalize(self.input.data)

                # Plotting
                if plot_period != None and ((i % plot_period == plot_period-1) or i == 0):
                    self.plot_input(i, plot_period, save=save, plot_inline=plot_inline)

                # External transforms
                for t in transforms:
                    self.input.data = t(self.input.data)

                # Cropping
                offset = _random_translation(offset, larger_shape[-2] - self.input_shape[-2], larger_shape[-1] - self.input_shape[-1], std=20)
                i0, i1 = int(offset[0]), int(offset[0])+self.input_shape[-2]
                j0, j1 = int(offset[1]), int(offset[1])+self.input_shape[-1]
                cropped_input = self.input[..., i0:i1, j0:j1]

                # Optimization
                self.model.zero_grad()
                self.model.forward(cropped_input.to(device))

                if objective.type == LAYER_OBJ:
                    loss = loss_func(layer_out)
                elif objective.type == CHANNEL_OBJ:
                    mask = torch.zeros_like(layer_out)
                    mask[0, objective.channel_index, :, :] = 1
                    loss = loss_func(mask * layer_out)
                elif objective.type == NEURON_OBJ:
                    mask = torch.zeros_like(layer_out)
                    mask[0][objective.neuron_index] = 1
                    loss = loss_func(mask * layer_out)

                loss += torch.mean(self.input**2) * reg

                loss_history[i] = loss

                loss.backward()
                optimizer.step()
                
            except KeyboardInterrupt:
                print("Stopped before iteration", i)
                break

        return start, self.input, loss_history[:i]