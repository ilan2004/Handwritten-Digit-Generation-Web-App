import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

# Set device (T4 GPU in Colab)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define Conditional VAE model
class CVAE(nn.Module):
    def __init__(self, latent_dim=20, num_classes=10):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # [32, 14, 14]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # [64, 7, 7]
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
        )
        self.label_emb = nn.Embedding(num_classes, 256)  # Embed digit labels
        self.fc_mu = nn.Linear(256 + 256, latent_dim)
        self.fc_logvar = nn.Linear(256 + 256, latent_dim)

        # Decoder
        self.decoder_input = nn.Linear(latent_dim + num_classes, 256)
        self.decoder = nn.Sequential(
            nn.Linear(256, 64 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # [32, 14, 14]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),  # [1, 28, 28]
            nn.Sigmoid(),
        )

    def encode(self, x, labels):
        h = self.encoder(x)
        label_embed = self.label_emb(labels)
        h = torch.cat([h, label_embed], dim=1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, labels):
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float()
        z = torch.cat([z, label_one_hot], dim=1)
        h = self.decoder_input(z)
        return self.decoder(h)

    def forward(self, x, labels):
        mu, logvar = self.encode(x, labels)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, labels), mu, logvar

# Loss function: Reconstruction loss (BCE) + KL divergence
def cvae_loss(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# Data loading
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# Initialize model, optimizer
model = CVAE(latent_dim=20, num_classes=10).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
num_epochs = 20
model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data, labels)
        loss = cvae_loss(recon_batch, data, mu, logvar)
        loss.backward()
        total_loss += loss.item()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_loader.dataset):.4f}')

# Save model
os.makedirs('models', exist_ok=True)
torch.save(model.state_dict(), 'models/cvae_mnist.pth')
print("Model saved to models/cvae_mnist.pth")