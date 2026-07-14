"""Generative domain base interfaces and models."""

from torch_like.models.generative.common import GenerativeBatch, GenerativeConfig, GenerativeModel
from torch_like.models.generative.controlnet import ControlNet, ControlNetConfig
from torch_like.models.generative.dcgan import DCGAN, DCGANConfig
from torch_like.models.generative.ddpm import DDPM, DDPMConfig
from torch_like.models.generative.dit import DiT, DiTConfig
from torch_like.models.generative.latent_diffusion import LatentDiffusion, LatentDiffusionConfig
from torch_like.models.generative.pixelcnn import PixelCNN, PixelCNNConfig
from torch_like.models.generative.realnvp import RealNVP, RealNVPConfig
from torch_like.models.generative.rectified_flow import RectifiedFlow, RectifiedFlowConfig
from torch_like.models.generative.vae import VAE, VAEConfig
from torch_like.models.generative.vqvae import VQVAE, VQVAEConfig
from torch_like.models.registry import register_model_catalog

model_zoo = {
    "VQVAE": {
        "cls": VQVAE,
        "family": "discrete_latent_generative_model",
        "layer": "Generative / vector-quantized image model",
        "integration_level": "paper_aligned_core",
        "supports": ["discrete_latent", "vector_quantization", "codebook", "reconstruction", "sampling"],
        "paper": "Neural Discrete Representation Learning (NeurIPS 2017)",
        "description": "VQ-VAE with convolutional encoder/decoder, vector quantization, codebook perplexity, and reconstruction loss",
        "complexity": "O(B * H * W * channels * encoder/decoder depth + B * latent_tokens * codebook_size * latent_dim)",
    },
    "PixelCNN": {
        "cls": PixelCNN,
        "family": "autoregressive_density_model",
        "layer": "Generative / autoregressive image model",
        "integration_level": "paper_aligned_core",
        "supports": ["autoregressive", "masked_convolution", "exact_likelihood", "sampling"],
        "paper": "Pixel Recurrent Neural Networks / PixelCNN (ICML 2016)",
        "description": "Masked-convolution autoregressive image model with exact discrete log likelihood and ancestral sampling",
        "complexity": "O(B * H * W * channels * num_values * layers) training; sampling is sequential over pixels",
    },
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
    "DiT": {
        "cls": DiT,
        "family": "diffusion_transformer",
        "layer": "Generative / image diffusion",
        "integration_level": "framework_oriented",
        "supports": ["diffusion", "denoising", "noise_prediction", "sampling", "transformer"],
        "paper": "Scalable Diffusion Models with Transformers (ICCV 2023)",
        "description": "DiT-style patch Transformer denoiser with timestep-conditioned blocks and diffusion loss",
        "complexity": "O(steps * B * P^2 * D) iterative denoising over image patch tokens",
    },
    "LatentDiffusion": {
        "cls": LatentDiffusion,
        "family": "latent_diffusion_model",
        "layer": "Generative / latent image diffusion",
        "integration_level": "framework_oriented",
        "supports": ["diffusion", "latent_space", "autoencoding", "noise_prediction", "sampling", "conditioning"],
        "paper": "High-Resolution Image Synthesis with Latent Diffusion Models (CVPR 2022)",
        "description": "Latent diffusion model with image encoder/decoder, latent-space denoising, optional vector conditioning, and image sampling",
        "complexity": "O(steps * B * H_latent * W_latent * channels * denoiser depth) plus encoder/decoder cost",
    },
    "ControlNet": {
        "cls": ControlNet,
        "family": "controlled_diffusion_model",
        "layer": "Generative / controllable image diffusion",
        "integration_level": "framework_oriented",
        "supports": ["diffusion", "latent_space", "control_conditioning", "zero_convolution", "sampling"],
        "paper": "Adding Conditional Control to Text-to-Image Diffusion Models (ICCV 2023)",
        "description": "ControlNet-style latent diffusion model with external control encoding and zero-initialized residual injection",
        "complexity": "O(steps * B * H_latent * W_latent * channels * denoiser depth + B * H * W * control encoder depth)",
    },
    "RectifiedFlow": {
        "cls": RectifiedFlow,
        "family": "flow_matching_model",
        "layer": "Generative / continuous flow model",
        "integration_level": "framework_oriented",
        "supports": ["flow_matching", "rectified_flow", "velocity_prediction", "euler_sampling", "conditioning"],
        "paper": "Flow Matching for Generative Modeling / Rectified Flow",
        "description": "Rectified Flow model that learns a velocity field from noise to data with interpolation, flow-matching loss, and Euler sampling",
        "complexity": "O(steps * B * H * W * channels * vector-field depth) Euler integration",
    },
    "RealNVP": {
        "cls": RealNVP,
        "family": "normalizing_flow",
        "layer": "Generative / density model",
        "integration_level": "framework_oriented",
        "supports": ["normalizing_flow", "exact_likelihood", "invertible", "sampling"],
        "paper": "Density Estimation using Real NVP (ICLR 2017)",
        "description": "Affine coupling normalizing flow with exact log likelihood and invertible sampling",
        "complexity": "O(B * layers * flattened_image_dim * hidden_dim) dense coupling flow",
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
    "DiTConfig",
    "DiT",
    "ControlNetConfig",
    "ControlNet",
    "LatentDiffusionConfig",
    "LatentDiffusion",
    "PixelCNNConfig",
    "PixelCNN",
    "RectifiedFlowConfig",
    "RectifiedFlow",
    "RealNVPConfig",
    "RealNVP",
    "VQVAEConfig",
    "VQVAE",
    "model_zoo",
]
