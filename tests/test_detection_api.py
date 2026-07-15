import unittest

import torch

import torch_like as tl
from torch_like.models import create_model
from torch_like.nn import CBAM1d


class DetectionApiTests(unittest.TestCase):
    def test_cbam1d_and_causal_tcn_detection_contract(self):
        attention = CBAM1d(channels=8, reduction=4, kernel_size=3)
        features = torch.randn(2, 8, 12)
        self.assertEqual(attention(features).shape, features.shape)

        model = create_model(
            "CBAMTCNDetector",
            domain="detection",
            input_dim=3,
            seq_len=10,
            d_model=8,
            e_layers=2,
            kernel_size=3,
            cbam_reduction=4,
            cbam_kernel_size=3,
        )
        x = torch.randn(2, 10, 3)
        batch = model.make_detection_batch(x)
        loss = model.fit_step(batch)
        score = model.score_batch(batch, reduction="none")
        loss.backward()
        self.assertEqual(model.predict(batch).shape, (2, 3))
        self.assertEqual(score.shape, (2, 3))

    def test_tranad_self_conditioned_detection_api(self):
        model = tl.TranAD(input_dim=3, seq_len=8, d_model=8, n_heads=2, e_layers=1, d_ff=16)
        batch = model.make_detection_batch(torch.randn(2, 8, 3))
        phase_1, phase_2 = model.reconstruct(batch.x)
        prediction = model.predict(batch)
        loss = model.fit_step(batch)
        score = model.score_batch(batch)
        loss.backward()
        self.assertEqual(phase_1.shape, (2, 8, 3))
        self.assertEqual(phase_2.shape, (2, 8, 3))
        self.assertEqual(prediction.shape, (2, 3))
        self.assertEqual(score.shape, (2,))

    def test_usad_dual_autoencoder_detection_api(self):
        model = create_model(
            "USAD",
            domain="detection",
            input_dim=3,
            seq_len=10,
            latent_dim=8,
            hidden_dim=16,
        )
        x = torch.randn(2, 10, 3)
        batch = model.make_detection_batch(x)
        w1, w2, w3 = model.reconstruct(x)
        prediction = model.predict(batch)
        loss = model.fit_step(batch)
        score = model.score_batch(batch)
        self.assertEqual(w1.shape, x.shape)
        self.assertEqual(w2.shape, x.shape)
        self.assertEqual(w3.shape, x.shape)
        self.assertEqual(prediction.shape, (2, 3))
        self.assertEqual(score.shape, (2,))
        self.assertEqual(loss.dim(), 0)

    def test_omni_anomaly_variational_recurrent_detection_api(self):
        model = create_model(
            "OmniAnomaly",
            domain="detection",
            input_dim=3,
            seq_len=10,
            d_model=8,
            latent_dim=4,
            hidden_dim=16,
            num_layers=1,
        )
        x = torch.randn(2, 10, 3)
        batch = model.make_detection_batch(x)
        reconstruction, mu, logvar = model.reconstruct(x)
        prediction = model.predict(batch)
        loss = model.fit_step(batch)
        score = model.score_batch(batch)
        self.assertEqual(reconstruction.shape, x.shape)
        self.assertEqual(mu.shape, (2, 10, 4))
        self.assertEqual(logvar.shape, (2, 10, 4))
        self.assertEqual(prediction.shape, (2, 3))
        self.assertEqual(score.shape, (2,))
        self.assertEqual(loss.dim(), 0)

    def test_anomaly_transformer_predict_api(self):
        cfg = tl.AnomalyTransformerConfig(input_dim=4, seq_len=10, d_model=16, n_heads=4, e_layers=1, d_ff=32)
        model = tl.AnomalyTransformer.from_config(cfg)
        x = torch.rand(3, cfg.seq_len, cfg.input_dim)
        batch = model.make_detection_batch(x)
        out = model.predict(batch)
        self.assertEqual(out.shape, (3, cfg.input_dim))

    def test_anomaly_transformer_score_batch(self):
        model = create_model(
            "AnomalyTransformer",
            domain="detection",
            input_dim=3,
            seq_len=10,
            d_model=15,
            n_heads=3,
            e_layers=1,
            d_ff=32,
        )
        x = torch.rand(2, 10, 3)
        batch = model.make_detection_batch(x)
        loss = model.fit_step(batch)
        score = model.score_batch(batch, reduction="none")
        self.assertEqual(loss.ndim, 0)
        self.assertEqual(score.shape, (2, 3))

    def test_dc_detector_predict_api(self):
        cfg = tl.DCDetectorConfig(input_dim=4, seq_len=10, d_model=16, n_heads=4, e_layers=1, d_ff=32)
        model = tl.DCDetector.from_config(cfg)
        x = torch.rand(3, cfg.seq_len, cfg.input_dim)
        batch = model.make_detection_batch(x)
        out = model.predict(batch)
        self.assertEqual(out.shape, (3, cfg.input_dim))

    def test_dc_detector_score_batch(self):
        model = create_model(
            "DCDetector",
            domain="detection",
            input_dim=3,
            seq_len=10,
            d_model=15,
            n_heads=3,
            e_layers=1,
            d_ff=32,
        )
        x = torch.rand(2, 10, 3)
        batch = model.make_detection_batch(x)
        loss = model.fit_step(batch)
        score = model.score_batch(batch, reduction="none")
        self.assertEqual(loss.ndim, 0)
        self.assertEqual(score.shape, (2, 3))

    def test_gdn_predict_api(self):
        cfg = tl.GDNConfig(input_dim=4, seq_len=10, d_model=16, top_k=2)
        model = tl.GDN.from_config(cfg)
        x = torch.rand(3, cfg.seq_len, cfg.input_dim)
        batch = model.make_detection_batch(x)
        out = model.predict(batch)
        self.assertEqual(out.shape, (3, cfg.input_dim))

    def test_gdn_score_batch(self):
        model = create_model(
            "GDN",
            domain="detection",
            input_dim=3,
            seq_len=10,
            d_model=16,
            top_k=2,
        )
        x = torch.rand(2, 10, 3)
        batch = model.make_detection_batch(x)
        loss = model.fit_step(batch)
        score = model.score_batch(batch, reduction="none")
        self.assertEqual(loss.ndim, 0)
        self.assertEqual(score.shape, (2, 3))

    def test_memto_predict_api(self):
        cfg = tl.MEMTOConfig(input_dim=4, seq_len=10, d_model=16, n_heads=4, e_layers=1, d_ff=32, mem_size=8)
        model = tl.MEMTO.from_config(cfg)
        x = torch.rand(3, cfg.seq_len, cfg.input_dim)
        batch = model.make_detection_batch(x)
        out = model.predict(batch)
        self.assertEqual(out.shape, (3, cfg.input_dim))

    def test_memto_score_batch(self):
        model = create_model(
            "MEMTO", domain="detection", input_dim=3, seq_len=10, d_model=16, n_heads=1, e_layers=1, d_ff=32, mem_size=8
        )
        x = torch.rand(2, 10, 3)
        batch = model.make_detection_batch(x)
        score = model.score_batch(batch, reduction="mean")
        self.assertEqual(score.shape, (2,))

    def test_mtadgat_predict_api(self):
        cfg = tl.MTADGATConfig(input_dim=4, seq_len=10, d_model=16, n_heads=4, e_layers=1, d_ff=32)
        model = tl.MTADGAT.from_config(cfg)
        x = torch.rand(3, cfg.seq_len, cfg.input_dim)
        batch = model.make_detection_batch(x)
        out = model.predict(batch)
        self.assertEqual(out.shape, (3, cfg.input_dim))

    def test_mtadgat_score_batch(self):
        model = create_model(
            "MTADGAT",
            domain="detection",
            input_dim=3,
            seq_len=10,
            d_model=15,
            n_heads=3,
            e_layers=1,
            d_ff=32,
        )
        x = torch.rand(2, 10, 3)
        batch = model.make_detection_batch(x)
        loss = model.fit_step(batch)
        score = model.score_batch(batch, reduction="none")
        self.assertEqual(loss.ndim, 0)
        self.assertEqual(score.shape, (2, 3))

    def test_rt_detector_predict_api(self):
        cfg = tl.RTDetectorConfig(input_dim=4, seq_len=10, ff_dim=16, projector_hidden_dim=8)
        model = tl.RTDetector.from_config(cfg)
        x = torch.rand(3, cfg.seq_len, cfg.input_dim)
        batch = model.make_detection_batch(x)
        out = model.predict(batch)
        self.assertEqual(out.shape, (3, cfg.input_dim))

    def test_rt_detector_anomaly_score(self):
        model = create_model("RTDetector", domain="detection", input_dim=3, seq_len=10)
        x = torch.rand(2, 10, 3)
        batch = model.make_detection_batch(x)
        score = model.anomaly_score(batch, reduction="mean")
        self.assertEqual(score.shape, (2,))

    def test_rt_detector_scores_remain_finite_under_distribution_shift(self):
        model = create_model("RTDetector", domain="detection", input_dim=3, seq_len=10)
        x = torch.randn(4, 10, 3) * 100
        score = model.score_batch(model.make_detection_batch(x), reduction="none")
        self.assertTrue(torch.isfinite(score).all())

    def test_detection_fit_step_and_score_batch_contract(self):
        model = create_model("RTDetector", domain="detection", input_dim=3, seq_len=10)
        x = torch.rand(2, 10, 3)
        batch = model.make_detection_batch(x)
        loss = model.fit_step(batch)
        score = model.score_batch(batch, reduction="none")
        self.assertEqual(loss.ndim, 0)
        self.assertEqual(score.shape, (2, 3))


if __name__ == "__main__":
    unittest.main()
