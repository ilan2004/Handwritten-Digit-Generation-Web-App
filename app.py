import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import numpy as np

# Define Conditional VAE model (same as in training script)
class CVAE(nn.Module):
    def __init__(self, latent_dim=20, num_classes=10):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
        )
        self.label_emb = nn.Embedding(num_classes, 256)
        self.fc_mu = nn.Linear(256 + 256, latent_dim)
        self.fc_logvar = nn.Linear(256 + 256, latent_dim)
        self.decoder_input = nn.Linear(latent_dim + num_classes, 256)
        self.decoder = nn.Sequential(
            nn.Linear(256, 64 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def decode(self, z, labels):
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float()
        z = torch.cat([z, label_one_hot], dim=1)
        h = self.decoder_input(z)
        return self.decoder(h)

# Load model
@st.cache_resource
def load_model():
    model = CVAE(latent_dim=20, num_classes=10)
    model.load_state_dict(torch.load('models/cvae_mnist.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

# Streamlit app
st.title("Handwritten Digit Generator")
st.write("Select a digit (0–9) to generate 5 handwritten images.")

# Digit selection
digit = st.selectbox("Choose a digit:", list(range(10)))

# Generate images
if st.button("Generate Images"):
    model = load_model()
    with torch.no_grad():
        # Generate 5 random latent vectors
        z = torch.randn(5, 20)
        # Create tensor for the selected digit
        labels = torch.full((5,), digit, dtype=torch.long)
        generated_images = model.decode(z, labels).cpu().numpy()

    # Display images
    st.subheader(f"Generated Images for Digit {digit}")
    cols = st.columns(5)
    for i in range(5):
        img = generated_images[i].squeeze() * 255
        img = Image.fromarray(img.astype(np.uint8))
        cols[i].image(img, caption=f"Image {i+1}", use_column_width=True)