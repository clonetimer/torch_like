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

    def test_dit_predict_loss_and_sample_api(self):
        model = tl.DiT(
            in_channels=1,
            image_size=8,
            d_model=16,
            num_timesteps=6,
            patch_size=4,
            n_heads=4,
            n_layers=1,
            d_ff=32,
            time_dim=16,
        )
        x_start = torch.randn(2, 1, 8, 8)
        timesteps = torch.tensor([0, 5])
        noise = torch.randn_like(x_start)
        x_noisy = model.q_sample(x_start, timesteps, noise)
        predicted = model.predict_noise(x_noisy, timesteps)
        loss = model.loss_step(model.make_generative_batch(x_start, timesteps, noise=noise))
        sample = model.sample(2, num_steps=2)
        self.assertEqual(predicted.shape, x_start.shape)
        self.assertEqual(sample.shape, x_start.shape)
        self.assertEqual(loss.dim(), 0)

    def test_dit_from_registry(self):
        model = create_model(
            "DiT",
            domain="generative",
            in_channels=1,
            image_size=8,
            d_model=16,
            num_timesteps=6,
            patch_size=4,
            n_heads=4,
            n_layers=1,
            d_ff=32,
            time_dim=16,
        )
        x_start = torch.randn(2, 1, 8, 8)
        timesteps = torch.tensor([0, 5])
        self.assertEqual(model.predict_noise(x_start, timesteps).shape, x_start.shape)

    def test_latent_diffusion_predict_loss_and_sample_api(self):
        model = tl.LatentDiffusion(
            in_channels=1,
            image_size=16,
            d_model=8,
            num_timesteps=6,
            latent_channels=2,
            downsample_factor=4,
            num_blocks=1,
            time_dim=16,
            condition_dim=3,
        )
        x_start = torch.randn(2, 1, 16, 16)
        timesteps = torch.tensor([0, 5])
        condition = torch.randn(2, 3)
        latents = model.encode_to_latent(x_start)
        noise = torch.randn_like(latents)
        z_noisy = model.q_sample(latents, timesteps, noise)
        predicted_noise = model.predict_noise(z_noisy, timesteps, condition=condition)
        reconstruction = model.decode_latent(latents)
        loss = model.loss_step(model.make_generative_batch(x_start, timesteps, noise=noise), condition=condition)
        sampled_latents = model.sample_latents(2, condition=condition, num_steps=2)
        sample = model.sample(2, condition=condition, num_steps=2)
        self.assertEqual(latents.shape, (2, 2, 4, 4))
        self.assertEqual(predicted_noise.shape, latents.shape)
        self.assertEqual(reconstruction.shape, x_start.shape)
        self.assertEqual(sampled_latents.shape, latents.shape)
        self.assertEqual(sample.shape, x_start.shape)
        self.assertEqual(loss.dim(), 0)

    def test_latent_diffusion_from_registry(self):
        model = create_model(
            "LatentDiffusion",
            domain="generative",
            in_channels=1,
            image_size=16,
            d_model=8,
            num_timesteps=6,
            latent_channels=2,
            downsample_factor=4,
            num_blocks=1,
            time_dim=16,
        )
        x_start = torch.randn(2, 1, 16, 16)
        timesteps = torch.tensor([0, 5])
        latents = model.encode_to_latent(x_start)
        self.assertEqual(model.predict_noise(latents, timesteps).shape, latents.shape)

    def test_controlnet_predict_loss_and_sample_api(self):
        model = tl.ControlNet(
            in_channels=1,
            image_size=16,
            d_model=8,
            num_timesteps=6,
            latent_channels=2,
            downsample_factor=4,
            num_blocks=1,
            time_dim=16,
            condition_dim=3,
            control_channels=1,
        )
        x_start = torch.randn(2, 1, 16, 16)
        control = torch.randn(2, 1, 16, 16)
        condition = torch.randn(2, 3)
        timesteps = torch.tensor([0, 5])
        latents = model.encode_to_latent(x_start)
        noise = torch.randn_like(latents)
        z_noisy = model.q_sample(latents, timesteps, noise)
        control_features = model.encode_control(control)
        residuals = model.control_residuals(control)
        predicted_noise = model.predict_noise(z_noisy, timesteps, control=control, condition=condition)
        loss = model.loss_step(
            model.make_generative_batch(x_start, timesteps, noise=noise), control=control, condition=condition
        )
        sampled_latents = model.sample_latents(control=control, condition=condition, num_steps=2)
        sample = model.sample(control=control, condition=condition, num_steps=2)
        self.assertEqual(control_features.shape, (2, 8, 4, 4))
        self.assertEqual(len(residuals), 1)
        self.assertEqual(residuals[0].shape, (2, 8, 4, 4))
        self.assertEqual(predicted_noise.shape, latents.shape)
        self.assertEqual(sampled_latents.shape, latents.shape)
        self.assertEqual(sample.shape, x_start.shape)
        self.assertEqual(loss.dim(), 0)

    def test_controlnet_from_registry(self):
        model = create_model(
            "ControlNet",
            domain="generative",
            in_channels=1,
            image_size=16,
            d_model=8,
            num_timesteps=6,
            latent_channels=2,
            downsample_factor=4,
            num_blocks=1,
            time_dim=16,
            control_channels=1,
        )
        x_start = torch.randn(2, 1, 16, 16)
        control = torch.randn(2, 1, 16, 16)
        timesteps = torch.tensor([0, 5])
        latents = model.encode_to_latent(x_start)
        self.assertEqual(model.predict_noise(latents, timesteps, control=control).shape, latents.shape)

    def test_rectified_flow_velocity_loss_and_sample_api(self):
        model = tl.RectifiedFlow(
            in_channels=1,
            image_size=16,
            d_model=8,
            num_timesteps=8,
            num_blocks=1,
            time_dim=16,
            condition_dim=3,
        )
        x1 = torch.randn(2, 1, 16, 16)
        x0 = torch.randn_like(x1)
        timesteps = torch.tensor([0.25, 0.75])
        condition = torch.randn(2, 3)
        x_t = model.interpolate(x0, x1, timesteps)
        target = model.target_velocity(x0, x1)
        predicted = model.predict_velocity(x_t, timesteps, condition=condition)
        loss = model.loss_step(model.make_generative_batch(x1, timesteps, noise=x0), condition=condition)
        sample = model.sample((2, 1, 16, 16), condition=condition, num_steps=2)
        self.assertEqual(x_t.shape, x1.shape)
        self.assertTrue(torch.allclose(target, x1 - x0))
        self.assertEqual(predicted.shape, x1.shape)
        self.assertEqual(sample.shape, x1.shape)
        self.assertEqual(loss.dim(), 0)

    def test_rectified_flow_from_registry(self):
        model = create_model(
            "RectifiedFlow",
            domain="generative",
            in_channels=1,
            image_size=16,
            d_model=8,
            num_timesteps=8,
            num_blocks=1,
            time_dim=16,
        )
        x = torch.randn(2, 1, 16, 16)
        timesteps = torch.tensor([0, 7])
        self.assertEqual(model.predict_velocity(x, timesteps).shape, x.shape)

    def test_realnvp_log_prob_sample_and_inverse_api(self):
        model = tl.RealNVP(
            in_channels=1,
            image_size=4,
            hidden_dim=16,
            num_coupling_layers=2,
        )
        x = torch.randn(2, 1, 4, 4)
        z, log_det = model(x)
        reconstruction = model.inverse(z)
        log_prob = model.log_prob(x)
        loss = model.loss_step(model.make_generative_batch(x, torch.zeros(2, dtype=torch.long)))
        sample = model.sample(2)
        self.assertEqual(z.shape, (2, 16))
        self.assertEqual(log_det.shape, (2,))
        self.assertEqual(log_prob.shape, (2,))
        self.assertEqual(reconstruction.shape, x.shape)
        self.assertTrue(torch.allclose(reconstruction, x, atol=1e-5))
        self.assertEqual(sample.shape, x.shape)
        self.assertEqual(loss.dim(), 0)

    def test_realnvp_from_registry(self):
        model = create_model(
            "RealNVP",
            domain="generative",
            in_channels=1,
            image_size=4,
            hidden_dim=16,
            num_coupling_layers=2,
        )
        x = torch.randn(2, 1, 4, 4)
        self.assertEqual(model.log_prob(x).shape, (2,))

    def test_vqvae_quantize_loss_and_sample_api(self):
        model = tl.VQVAE(
            in_channels=1,
            image_size=16,
            latent_dim=8,
            num_embeddings=16,
            hidden_channels=(8, 16),
        )
        x = torch.randn(2, 1, 16, 16)
        batch = model.make_generative_batch(x, torch.zeros(2, dtype=torch.long))
        out = model.predict(batch)
        loss = model.loss_step(batch)
        sample = model.sample(2)
        self.assertEqual(out["reconstruction"].shape, x.shape)
        self.assertEqual(out["z"].shape, (2, 8, 4, 4))
        self.assertEqual(out["z_q"].shape, (2, 8, 4, 4))
        self.assertEqual(out["indices"].shape, (2, 4, 4))
        self.assertEqual(sample.shape, x.shape)
        self.assertEqual(loss.dim(), 0)
        self.assertEqual(out["perplexity"].dim(), 0)

    def test_vqvae_from_registry(self):
        model = create_model(
            "VQVAE",
            domain="generative",
            in_channels=1,
            image_size=16,
            latent_dim=8,
            num_embeddings=16,
            hidden_channels=(8, 16),
        )
        x = torch.randn(2, 1, 16, 16)
        z = model.encode(x)
        z_q, _, indices, _ = model.quantize(z)
        self.assertEqual(model.decode(z_q).shape, x.shape)
        self.assertEqual(indices.shape, (2, 4, 4))

    def test_pixelcnn_log_prob_loss_and_sample_api(self):
        model = tl.PixelCNN(
            in_channels=1,
            image_size=4,
            num_values=8,
            hidden_channels=8,
            num_layers=2,
            kernel_size=3,
        )
        x = torch.randint(0, 8, (2, 1, 4, 4)).float()
        batch = model.make_generative_batch(x, torch.zeros(2, dtype=torch.long))
        logits = model.predict(batch)
        loss = model.loss_step(batch)
        log_prob = model.log_prob(x)
        sample = model.sample(2)
        self.assertEqual(logits.shape, (2, 1, 8, 4, 4))
        self.assertEqual(log_prob.shape, (2,))
        self.assertEqual(sample.shape, x.shape)
        self.assertEqual(loss.dim(), 0)

    def test_pixelcnn_from_registry(self):
        model = create_model(
            "PixelCNN",
            domain="generative",
            in_channels=1,
            image_size=4,
            num_values=8,
            hidden_channels=8,
            num_layers=2,
            kernel_size=3,
        )
        x = torch.randint(0, 8, (2, 1, 4, 4)).float()
        self.assertEqual(model.log_prob(x).shape, (2,))


if __name__ == "__main__":
    unittest.main()
