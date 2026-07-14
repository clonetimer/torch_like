import unittest

import torch

import torch_like as tl
from torch_like.models import create_model


class AudioApiTests(unittest.TestCase):
    def test_encodec_rvq_bandwidth_reconstruction_and_registry_api(self):
        rvq = tl.ResidualVectorQuantizer(dim=8, codebook_size=16, num_codebooks=3)
        latent = torch.randn(2, 7, 8)
        quantized, codes, commitment = rvq(latent, num_codebooks=2)
        self.assertEqual(quantized.shape, latent.shape)
        self.assertEqual(codes.shape, (2, 2, 7))
        self.assertEqual(rvq.decode(codes).shape, latent.shape)
        self.assertEqual(commitment.dim(), 0)

        model = create_model(
            "EnCodec", domain="audio", input_dim=1, output_dim=1, d_model=8,
            sample_rate=8000, channels=(4, 8), strides=(2, 4), codebook_size=16,
            num_codebooks=4, target_bandwidth=4.0,
        )
        waveform = torch.randn(2, 129)
        encoded = model.encode(waveform, bandwidth=2.0)
        reconstruction = model.predict(model.make_audio_batch(waveform))
        decoded = model.decode(encoded["codes"], length=waveform.size(-1))
        loss = model.loss_step(model.make_audio_batch(waveform))
        loss.backward()
        self.assertEqual(encoded["codes"].dim(), 3)
        self.assertLessEqual(encoded["codes"].size(1), 4)
        self.assertEqual(reconstruction.shape, waveform.shape)
        self.assertEqual(decoded.shape, waveform.shape)
        self.assertEqual(loss.dim(), 0)
        self.assertIsNotNone(model.encoder[0].weight.grad)
        self.assertGreater(model.encoder[0].weight.grad.norm().item(), 0.0)

    def test_wav2vec2_raw_waveform_api(self):
        model = tl.Wav2Vec2(
            input_dim=1,
            output_dim=8,
            d_model=16,
            feature_channels=(4, 8),
            kernel_sizes=(5, 3),
            strides=(2, 2),
            n_heads=4,
            n_layers=1,
            d_ff=32,
        )
        waveform = torch.randn(2, 64)
        features = model.extract_features(waveform)
        encoded = model.encode(waveform)
        logits = model.predict(model.make_audio_batch(waveform))
        self.assertEqual(features.shape[-1], 16)
        self.assertEqual(encoded.shape, features.shape)
        self.assertEqual(logits.shape[:2], encoded.shape[:2])
        self.assertEqual(logits.shape[-1], 8)

    def test_wav2vec2_from_registry(self):
        model = create_model(
            "Wav2Vec2",
            domain="audio",
            output_dim=8,
            d_model=16,
            feature_channels=(4, 8),
            kernel_sizes=(5, 3),
            strides=(2, 2),
            n_heads=4,
            n_layers=1,
            d_ff=32,
        )
        self.assertEqual(model(torch.randn(2, 64)).shape[-1], 8)

    def test_hubert_masked_hidden_unit_prediction_api(self):
        model = tl.HuBERT(
            input_dim=1,
            output_dim=8,
            d_model=16,
            feature_channels=(4, 8),
            kernel_sizes=(5, 3),
            strides=(2, 2),
            n_heads=4,
            n_layers=1,
            d_ff=32,
            mask_prob=0.25,
            mask_length=2,
        )
        waveform = torch.randn(2, 64)
        features = model.extract_features(waveform)
        mask = torch.zeros(features.shape[:2], dtype=torch.bool)
        mask[:, :2] = True
        masked, used_mask = model.mask_features(features, mask)
        encoded = model.encode(waveform, mask=used_mask)
        logits = model.predict_units(waveform, mask=used_mask)
        labels = torch.randint(0, 8, features.shape[:2])
        loss = model.loss_step(model.make_audio_batch(waveform, y=labels))
        loss.backward()
        self.assertEqual(features.shape[-1], 16)
        self.assertEqual(masked.shape, features.shape)
        self.assertEqual(used_mask.shape, features.shape[:2])
        self.assertEqual(encoded.shape, features.shape)
        self.assertEqual(logits.shape[:2], features.shape[:2])
        self.assertEqual(logits.shape[-1], 8)
        self.assertEqual(loss.dim(), 0)

    def test_hubert_from_registry(self):
        model = create_model(
            "HuBERT",
            domain="audio",
            output_dim=8,
            d_model=16,
            feature_channels=(4, 8),
            kernel_sizes=(5, 3),
            strides=(2, 2),
            n_heads=4,
            n_layers=1,
            d_ff=32,
            mask_length=2,
        )
        self.assertEqual(model(torch.randn(2, 64)).shape[-1], 8)

    def test_conformer_sequence_logits_and_loss_api(self):
        model = tl.Conformer(input_dim=6, output_dim=5, d_model=16, n_heads=4, n_layers=1, d_ff=32, conv_kernel_size=3)
        x = torch.randn(2, 7, 6)
        y = torch.randint(0, 5, (2, 7))
        lengths = torch.tensor([7, 5])
        batch = model.make_audio_batch(x, lengths=lengths, y=y)
        encoded = model.encode(x, lengths=lengths)
        logits = model.predict(batch)
        loss = model.loss_step(batch)
        loss.backward()
        self.assertEqual(encoded.shape, (2, 7, 16))
        self.assertEqual(logits.shape, (2, 7, 5))
        self.assertEqual(loss.dim(), 0)

    def test_conformer_from_registry(self):
        model = create_model(
            "Conformer",
            domain="audio",
            input_dim=6,
            output_dim=5,
            d_model=16,
            n_heads=4,
            n_layers=1,
            d_ff=32,
            conv_kernel_size=3,
        )
        self.assertEqual(model(torch.randn(2, 7, 6)).shape, (2, 7, 5))

    def test_whisper_encoder_decoder_generation_and_loss_api(self):
        model = tl.Whisper(
            input_dim=8,
            output_dim=16,
            d_model=16,
            max_audio_positions=8,
            max_text_positions=8,
            n_heads=4,
            encoder_layers=1,
            decoder_layers=1,
            d_ff=32,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
        x = torch.randn(2, 8, 12)
        lengths = torch.tensor([12, 8])
        token_ids = torch.tensor([[1, 4, 5, 2, 0], [1, 7, 3, 2, 0]])
        batch = model.make_audio_batch(x, lengths=lengths, y=token_ids)
        encoded = model.encode_audio(x, lengths=lengths)
        logits = model(x, lengths=lengths, decoder_input_ids=token_ids[:, :-1])
        predicted = model.predict(batch)
        loss = model.loss_step(batch)
        generated = model.generate(x, lengths=lengths, max_new_tokens=3)
        self.assertEqual(encoded.shape, (2, 6, 16))
        self.assertEqual(logits.shape, (2, 4, 16))
        self.assertEqual(predicted.shape, (2, 1, 16))
        self.assertEqual(generated.shape[0], 2)
        self.assertLessEqual(generated.shape[1], 4)
        self.assertEqual(loss.dim(), 0)

    def test_whisper_from_registry(self):
        model = create_model(
            "Whisper",
            domain="audio",
            input_dim=8,
            output_dim=16,
            d_model=16,
            max_audio_positions=8,
            max_text_positions=8,
            n_heads=4,
            encoder_layers=1,
            decoder_layers=1,
            d_ff=32,
        )
        logits = model(torch.randn(2, 8, 12), decoder_input_ids=torch.ones(2, 3, dtype=torch.long))
        self.assertEqual(logits.shape, (2, 3, 16))

    def test_audio_spectrogram_transformer_api(self):
        model = tl.AudioSpectrogramTransformer(
            input_dim=1,
            output_dim=4,
            d_model=16,
            freq_bins=16,
            time_frames=16,
            patch_size=(4, 4),
            n_heads=4,
            n_layers=1,
            d_ff=32,
        )
        x = torch.randn(2, 1, 16, 16)
        y = torch.tensor([0, 1])
        batch = model.make_audio_batch(x, y=y)
        features = model.forward_features(x)
        logits = model.predict(batch)
        loss = model.loss_step(batch)
        self.assertEqual(features.shape, (2, 16))
        self.assertEqual(logits.shape, (2, 4))
        self.assertEqual(loss.dim(), 0)

    def test_audio_spectrogram_transformer_from_registry(self):
        model = create_model(
            "AudioSpectrogramTransformer",
            domain="audio",
            input_dim=1,
            output_dim=4,
            d_model=16,
            freq_bins=16,
            time_frames=16,
            patch_size=(4, 4),
            n_heads=4,
            n_layers=1,
            d_ff=32,
        )
        self.assertEqual(model(torch.randn(2, 1, 16, 16)).shape, (2, 4))


if __name__ == "__main__":
    unittest.main()
