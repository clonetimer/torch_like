import unittest

import torch

import torch_like as tl
from torch_like.models import create_model


class DetectionApiTests(unittest.TestCase):
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
        model = create_model("MEMTO", domain="detection", input_dim=3, seq_len=10, d_model=16, n_heads=1, e_layers=1, d_ff=32, mem_size=8)
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
