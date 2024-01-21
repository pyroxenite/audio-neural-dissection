import torch
import torch.nn as nn

########## DCGAN ##########

def train_dcgan(generator, generator_optimizer, discriminator, discriminator_optimizer, n_epochs, train_loader, device):
    
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

########## WGAN ##########

def _gradient_penalty(gradient):
    gradient = gradient.view(len(gradient), -1)
    gradient_norm = gradient.norm(2, dim=1)
    penalty = torch.mean((gradient_norm-1)**2)
    return penalty

def _get_gen_loss(fake_pred):
    gen_loss = -1*torch.mean(fake_pred)
    return gen_loss

def _get_disc_loss(fake_pred, real_pred, penalty, c_lambda):
    disc_loss = torch.mean(fake_pred) - torch.mean(real_pred) + c_lambda*penalty
    return disc_loss

def _get_gradient(discriminator, real_samples, fake_samples, eps):
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

def train_wgan(generator, generator_optimizer, discriminator, discriminator_optimizer, n_epochs, z_dim, train_loader, device):
    
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
                discriminator_optimizer.zero_grad()
                
                disc_real_pred = discriminator(real_samples)
                noise = torch.randn(curr_batch_size, z_dim, device=device)
                fake_images = generator(noise)
                disc_fake_pred = discriminator(fake_images)
                
                eps = torch.rand(curr_batch_size, 1, 1, 1, device=device, requires_grad=True)
                gradient = _get_gradient(discriminator, real_samples, fake_images.detach(), eps)
                grad_penalty = _gradient_penalty(gradient)
                
                disc_loss = _get_disc_loss(disc_fake_pred, disc_real_pred, grad_penalty, c_lambda)
                
                disc_loss.backward(retain_graph=True)
                discriminator_optimizer.step()
                
                mean_disc_loss += disc_loss.item() / n_train_disc
                
            ### Training the generator
            generator_optimizer.zero_grad()
            noise = torch.randn(curr_batch_size, z_dim, device=device)
            fake_images = generator(noise)
            disc_fake_pred = discriminator(fake_images)
            
            gen_loss = _get_gen_loss(disc_fake_pred)
            
            gen_loss.backward()
            generator_optimizer.step()

            if n%n_print == 0:
                print(f"> [Epoch {epoch+1}] Batch n°{n} ---- | Disc_loss : {mean_disc_loss:.04f} || Gen_loss : {gen_loss.item():.04f} |")
        
        torch.save(generator, "code/models/WGAN/backup/mnist-wgan-generator.pt")
        torch.save(discriminator, "code/models/WGAN/backup/mnist-wgan-discriminator.pt")

    print("\n### Done Training ###\n")
    
    return generator, discriminator