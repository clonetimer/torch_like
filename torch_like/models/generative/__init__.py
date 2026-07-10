"""Generative domain base interfaces and models."""

from torch_like.models.generative.common import GenerativeBatch, GenerativeConfig, GenerativeModel
from torch_like.models.generative.dcgan import DCGAN, DCGANConfig
from torch_like.models.generative.ddpm import DDPM, DDPMConfig
from torch_like.models.generative.vae import VAE, VAEConfig
from torch_like.models.registry import register_model_catalog

model_zoo = {
    "DDPM": {
        "cls": DDPM,
        "family": "diffusion_model",
        "layer": "Generative / image diffusion",
        "integration_level": "framework_oriented",
        "supports": ["diffusion", "denoising", "noise_prediction", "sampling"],
        "paper": "Denoising Diffusion Probabilistic Models (NeurIPS 2020)",
        "description": "DDPM-style image diffusion model with forward noising, epsilon prediction, loss_step, and iterative sampling",
        "complexity": "O(steps * B * H * W * channels * denoiser depth) iterative denoising",
    },
    "VAE": {
        "cls": VAE,
        "family": "variational_autoencoder",
        "layer": "Generative / latent variable image model",
        "integration_level": "framework_oriented",
        "supports": ["latent_variable", "reparameterization", "elbo_loss", "sampling"],
        "paper": "Auto-Encoding Variational Bayes (ICLR 2014)",
        "description": "Convolutional VAE with encoder statistics, reparameterized latent samples, reconstruction, and ELBO-style loss",
        "complexity": "O(B * H * W * channels * encoder/decoder depth) latent-variable image generator",
    },
    "DCGAN": {
        "cls": DCGAN,
        "family": "adversarial_generative_model",
        "layer": "Generative / image adversarial model",
        "integration_level": "framework_oriented",
        "supports": ["adversarial_loss", "generator", "discriminator", "sampling"],
        "paper": "Generative Adversarial Nets (NeurIPS 2014) / Unsupervised Representation Learning with DCGANs (ICLR 2016)",
        "description": "DCGAN-style image generator/discriminator pair with sampling and adversarial loss helpers",
        "complexity": "O(B * H * W * channels * generator/discriminator depth) adversarial image generator",
    },
}

register_model_catalog("generative", model_zoo)

__all__ = [
    "GenerativeConfig",
    "GenerativeBatch",
    "GenerativeModel",
    "DDPMConfig",
    "DDPM",
    "VAEConfig",
    "VAE",
    "DCGANConfig",
    "DCGAN",
    "model_zoo",
]
