import unittest

import torch

import torch_like as tl
from torch_like.models import create_model


class GenerativeApiTests(unittest.TestCase):
    def test_ddpm_predict_and_loss_api(self):
        model = tl.DDPM(
            in_channels=1,
            image_size=16,
            d_model=8,
            num_timesteps=8,
            num_blocks=1,
            time_dim=16,
        )
        x_start = torch.randn(2, 1, 16, 16)
        timesteps = torch.randint(0, model.num_timesteps, (2,))
        noise = torch.randn_like(x_start)
        x_noisy = model.q_sample(x_start, timesteps, noise)
        batch = model.make_generative_batch(x_start, timesteps, noise=noise)
        predicted_noise = model.predict(model.make_generative_batch(x_noisy, timesteps))
        loss = model.loss_step(batch)
        sample = model.sample((2, 1, 16, 16), num_steps=2)
        self.assertEqual(x_noisy.shape, x_start.shape)
        self.assertEqual(predicted_noise.shape, x_start.shape)
        self.assertEqual(sample.shape, x_start.shape)
        self.assertEqual(loss.dim(), 0)
        self.assertTrue(torch.isfinite(sample).all())
        with self.assertRaises(ValueError):
            model.sample((2, 1, 16, 16), num_steps=0)

    def test_ddpm_from_registry(self):
        model = create_model(
            "DDPM",
            domain="generative",
            in_channels=1,
            image_size=16,
            d_model=8,
            num_timesteps=6,
            num_blocks=1,
            time_dim=16,
        )
        x_start = torch.randn(2, 1, 16, 16)
        timesteps = torch.tensor([0, 5])
        predicted_noise = model.predict_noise(model.q_sample(x_start, timesteps), timesteps)
        self.assertEqual(predicted_noise.shape, x_start.shape)

    def test_vae_predict_loss_and_sample_api(self):
        model = tl.VAE(
            in_channels=1,
            image_size=16,
            latent_dim=4,
            hidden_channels=(4, 8),
        )
        x_start = torch.randn(2, 1, 16, 16)
        batch = model.make_generative_batch(x_start, torch.zeros(2, dtype=torch.long))
        out = model.predict(batch)
        loss = model.loss_step(batch)
        sample = model.sample(2)
        self.assertEqual(out["reconstruction"].shape, x_start.shape)
        self.assertEqual(out["mu"].shape, (2, 4))
        self.assertEqual(out["logvar"].shape, (2, 4))
        self.assertEqual(out["z"].shape, (2, 4))
        self.assertEqual(sample.shape, x_start.shape)
        self.assertEqual(loss.dim(), 0)

    def test_vae_from_registry(self):
        model = create_model(
            "VAE",
            domain="generative",
            in_channels=1,
            image_size=16,
            latent_dim=4,
            hidden_channels=(4, 8),
        )
        x_start = torch.randn(2, 1, 16, 16)
        mu, logvar = model.encode(x_start)
        reconstruction = model.decode(model.reparameterize(mu, logvar))
        self.assertEqual(mu.shape, (2, 4))
        self.assertEqual(logvar.shape, (2, 4))
        self.assertEqual(reconstruction.shape, x_start.shape)

    def test_dcgan_sample_and_loss_api(self):
        model = tl.DCGAN(
            in_channels=1,
            image_size=16,
            latent_dim=4,
            feature_maps=4,
        )
        z = model.sample_latent(2)
        fake = model(z)
        real = torch.randn(2, 1, 16, 16)
        logits = model.discriminate(real)
        gen_loss = model.generator_loss(fake)
        disc_loss = model.discriminator_loss(real, fake)
        batch_losses = model.loss_step(model.make_generative_batch(real, torch.zeros(2, dtype=torch.long)))
        self.assertEqual(z.shape, (2, 4))
        self.assertEqual(fake.shape, real.shape)
        self.assertEqual(logits.shape, (2,))
        self.assertEqual(gen_loss.dim(), 0)
        self.assertEqual(disc_loss.dim(), 0)
        self.assertEqual(batch_losses["generator_loss"].dim(), 0)
        self.assertEqual(batch_losses["discriminator_loss"].dim(), 0)

    def test_dcgan_from_registry(self):
        model = create_model(
            "DCGAN",
            domain="generative",
            in_channels=1,
            image_size=16,
            latent_dim=4,
            feature_maps=4,
        )
        sample = model.sample(2)
        logits = model.discriminate(sample)
        self.assertEqual(sample.shape, (2, 1, 16, 16))
        self.assertEqual(logits.shape, (2,))

    def test_dcgan_rejects_unsupported_image_size(self):
        with self.assertRaises(ValueError):
            tl.DCGAN(in_channels=1, image_size=4, latent_dim=4, feature_maps=4)


if __name__ == "__main__":
    unittest.main()
