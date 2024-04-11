import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import os
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VAE(nn.Module):
    def __init__(self, latent_dim=10):
        super(VAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.fc_mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(64 * 7 * 7, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64*7*7),
            nn.ReLU(),
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    def forward(self, x, return_latent=False):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)
        if return_latent:
            return z
        else:
            return self.decoder(z), mu, logvar

class BaseModel():
    def __init__(self, args):
        super(BaseModel, self).__init__()
        
        self.device = device
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.latent_dim = args.latent_dim
        self.lr = args.lr
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.noise_level = args.noise_level
        
        self.test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=self.transform, download=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=7, shuffle=True)

    @staticmethod
    def loss_function(recon_x, x, mu, logvar):
        BCE = torch.nn.functional.binary_cross_entropy(recon_x.view(-1, 784), x.view(-1, 784), reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD

    def train(self, model):
        train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=self.transform, download=True)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = optim.Adam(model.parameters(), lr=self.lr)

        for epoch in range(self.epochs):
            for images, _ in train_loader:
                images = images.to(self.device)
        
                optimizer.zero_grad()
                reconstructed_images, mu, logvar = model(images)
                loss = self.loss_function(reconstructed_images, images, mu, logvar)
                loss.backward()
                optimizer.step()
            
            print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {loss.item():.4f}")
        
        torch.save(model.state_dict(), 'vae_model.pth')

        print('Finished Training')
    
    @staticmethod
    def add_noise(images, noise_level):
        noise = torch.randn_like(images) - 0.5
        noisy_images = torch.clamp(images + 0.5 * noise_level, 0, 1)
        return noisy_images

    @staticmethod
    def save_output(plt, filename):
        if not os.path.exists('output'):
            os.makedirs('output')
        if not os.path.exists(os.path.join('output', 'vae')):
            os.makedirs(os.path.join('output', 'vae'))
        plt.savefig(os.path.join('output/vae', filename))
    
    def evaluate(self, model):
        test_images, _ = next(iter(self.test_loader))
        test_images = test_images.to(self.device)

        reconstructed_images, _, _ = model(test_images)

        fig, axes = plt.subplots(nrows=2, ncols=7, figsize=(10, 6))

        for images, row in zip([test_images, reconstructed_images], axes):
            for img, ax in zip(images, row):
                ax.imshow(img[0].cpu().detach().numpy().squeeze(), cmap='gray')  # Select the first image from the batch
                ax.axis('off')

        axes[0, 0].set_title('Original')
        axes[1, 0].set_title('Reconstructed')

        self.save_output(plt, 'evaluation.png')

        plt.tight_layout()
        plt.show()
        
    def reconstruc_noisy_images(self, model):
        test_images, _ = next(iter(self.test_loader))
        test_images = test_images.to(self.device)

        noisy_images = self.add_noise(test_images, noise_level=0.5)

        reconstructed_noisy_images, _, _ = model(noisy_images)

        fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(10, 6))

        for images, row in zip([test_images, noisy_images, reconstructed_noisy_images], axes):
            for img, ax in zip(images, row):
                ax.imshow(img[0].cpu().detach().numpy().squeeze(), cmap='gray')  # Select the first image from the batch
                ax.axis('off')

        axes[0, 0].set_title('Original')
        axes[1, 0].set_title('Noisy')
        axes[2, 0].set_title('Reconstructed (Noisy)')

        self.save_output(plt, 'reconstructed-noisy.png')

        plt.tight_layout()
        plt.show()
    
    def sample_latent_space(self, model):
        z = torch.randn(30, self.latent_dim, device=self.device)
        
        with torch.no_grad():
            fake_images = model.decoder(z).cpu()

        fig, axes = plt.subplots(nrows=3, ncols=10, figsize=(15, 5))
        fig.suptitle("Sample Latent Space")
        for i, ax in enumerate(axes.flatten()):
            ax.imshow(fake_images[i].cpu().detach().numpy().squeeze(), cmap='gray')
            ax.axis('off')
        
        self.save_output(plt, 'sample_latent_space.png')

        plt.tight_layout()
        plt.show()
    
    def interpolate_latent(self, model, alpha=0.5):
        test_images, _ = next(iter(self.test_loader))
        test_images = test_images.to(self.device)
        image1, image2 = test_images[0:1], test_images[1:2]

        z1 = model(image1, return_latent=True)
        z2 = model(image2, return_latent=True)

        z = (1 - alpha) * z1 + alpha * z2

        with torch.no_grad():
            fake_images = model.decoder(z).cpu()

        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
        axes[0].imshow(image1.cpu().detach().numpy().squeeze(), cmap='gray')
        axes[0].set_title('Image 1')
        axes[1].imshow(image2.cpu().detach().numpy().squeeze(), cmap='gray')
        axes[1].set_title('Image 2')
        axes[2].imshow(fake_images[0].cpu().detach().numpy().squeeze(), cmap='gray')
        axes[2].set_title('Interpolated Image')
        for ax in axes:
            ax.axis('off')
        
        self.save_output(plt, 'interpolate_latent.png')

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AutoEncoder Modes')
    parser.add_argument('--mode', type=str, required=True, 
                        help='Mode to run the AutoEncoder. Options: "train", "evaluate", "noisy", "sampling", "interpolate"')
    parser.add_argument('--model', type=str, default='vae_model.pth',
                        help='Path to the trained model.')
    parser.add_argument('--alpha', type=float, default=0.5, 
                        help='Alpha value for interpolation. Only used in "interpolate" mode.')
    parser.add_argument('--noise_level', type=float, default=0.5,
                        help='Noise level for adding noise to the images.')
    parser.add_argument('--latent_dim', type=int, default=10, 
                        help='Latent dimension for the AutoEncoder model.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate for the AutoEncoder model.')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs for training the AutoEncoder model.')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training the AutoEncoder model.')
    args = parser.parse_args()

    model = BaseModel(args)

    print("-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-\nUsing VAE AutoEncoder\n-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-")
    auto_encoder_model = VAE(args.latent_dim).to(device)

    mode = args.mode

    if mode != 'train':
        auto_encoder_model.load_state_dict(torch.load(args.model))

    if mode == 'train':
        model.train(auto_encoder_model)
    elif mode == 'evaluate':
        model.evaluate(auto_encoder_model)
    elif mode == 'noisy':
        model.reconstruc_noisy_images(auto_encoder_model)
    elif mode == 'sampling':
        model.sample_latent_space(auto_encoder_model)
    elif mode == 'interpolate':
        model.interpolate_latent(auto_encoder_model, alpha=args.alpha)
    else:
        print(f'Invalid mode: {mode}. Options are "train", "evaluate", "noisy", "sampling", "interpolate".')