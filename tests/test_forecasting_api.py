import unittest

import torch
import numpy as np

from torch_like.models import create_model
from torch_like.nn.attention import AutoCorrelation
from torch_like.models.thgnn import THGNN
from torch_like.models.forecasting import (
    AGCRN,
    AST,
    ASTGCN,
    ConvTimeNet,
    Crossformer,
    DAG,
    D2STGNN,
    DCRNN,
    DeepAR,
    DLinear,
    DMSTGCN,
    FEDformer,
    FCGAGA,
    FreTS,
    GTS,
    ForecastingConfig,
    GraphWaveNet,
    Informer,
    ITransformer,
    Koopa,
    InformerRUL,
    LogTrans,
    MICN,
    MLCNN,
    MTGNN,
    NBEATS,
    NHITS,
    PatchTST,
    PPM,
    ProTran,
    SCINet,
    SOFTS,
    RULConfig,
    STDN,
    STFGNN,
    STGODE,
    StemGNN,
    TFT,
    TimeGrad,
    TimeMixer,
    TSMixer,
    TimesNet,
    TiDE,
    TimeXer,
)


class ForecastingApiTests(unittest.TestCase):
    def test_make_forecast_batch_adapts_different_decoder_dimension(self):
        model = Informer(
            enc_in=4,
            dec_in=2,
            c_out=2,
            seq_len=12,
            label_len=4,
            pred_len=3,
            d_model=8,
            n_heads=2,
            e_layers=1,
            d_layers=1,
            d_ff=16,
        )
        x_enc = torch.randn(2, 12, 4)
        inferred = model.make_forecast_batch(x_enc)
        self.assertEqual(inferred.x_dec.shape, (2, 7, 2))
        self.assertTrue(torch.equal(inferred.x_dec[:, :4], x_enc[:, -4:, -2:]))
        batch = model.make_forecast_batch(x_enc, x_target_enc=x_enc[..., :2])
        self.assertEqual(batch.x_dec.shape, (2, 7, 2))
        self.assertEqual(model.predict(batch).shape, (2, 3, 2))

    def test_make_forecast_batch_handles_zero_label_length(self):
        model = Informer(
            enc_in=2,
            dec_in=2,
            c_out=2,
            seq_len=8,
            label_len=0,
            pred_len=3,
            d_model=8,
            n_heads=2,
            e_layers=1,
            d_layers=1,
            d_ff=16,
        )
        batch = model.make_forecast_batch(torch.randn(2, 8, 2))
        self.assertEqual(batch.x_dec.shape, (2, 3, 2))

    def test_autocorrelation_respects_explicit_mask(self):
        attention = AutoCorrelation(mask_flag=True, factor=1, attention_dropout=0.0)
        queries = torch.randn(1, 4, 1, 2)
        keys = torch.randn(1, 4, 1, 2)
        values = torch.randn(1, 4, 1, 2)
        denied = torch.zeros(1, 1, 4, 4, dtype=torch.bool)
        out, _ = attention(queries, keys, values, attn_mask=denied)
        self.assertTrue(torch.equal(out, torch.zeros_like(out)))

    def test_lazy_model_param_count_before_and_after_initialization(self):
        model = TiDE(
            enc_in=4,
            dec_in=2,
            c_out=2,
            seq_len=12,
            label_len=6,
            pred_len=3,
            d_model=16,
            n_heads=4,
            e_layers=1,
            d_layers=1,
            d_ff=32,
        )
        before = model.param_count()
        x = torch.randn(2, 12, 4)
        model.predict(model.make_forecast_batch(x))
        self.assertGreater(model.param_count(), before)

    def test_forecasting_predict_api(self):
        cfg = ForecastingConfig(
            enc_in=2,
            dec_in=2,
            c_out=2,
            seq_len=16,
            label_len=8,
            pred_len=4,
            d_model=32,
            n_heads=4,
            e_layers=1,
            d_layers=1,
            d_ff=64,
        )
        model = create_model("Informer", domain="forecasting", **cfg.__dict__)
        x_enc = torch.randn(3, cfg.seq_len, cfg.enc_in)
        batch = model.make_forecast_batch(x_enc)
        out = model.predict(batch)
        self.assertEqual(out.shape, (3, cfg.pred_len, cfg.c_out))

    def test_model_from_config_roundtrip(self):
        cfg = ForecastingConfig(
            enc_in=1,
            dec_in=1,
            c_out=1,
            seq_len=12,
            label_len=6,
            pred_len=3,
            d_model=16,
            n_heads=4,
            e_layers=1,
            d_layers=1,
            d_ff=32,
        )
        model = LogTrans.from_config(cfg)
        x_enc = torch.randn(2, cfg.seq_len, cfg.enc_in)
        batch = model.make_forecast_batch(x_enc)
        out = model.predict(batch)
        self.assertEqual(out.shape, (2, cfg.pred_len, cfg.c_out))

    def test_fedformer_predict_api(self):
        cfg = ForecastingConfig(
            enc_in=3,
            dec_in=3,
            c_out=3,
            seq_len=16,
            label_len=8,
            pred_len=4,
            d_model=16,
            n_heads=4,
            e_layers=2,
            d_layers=1,
            d_ff=32,
            dropout=0.1,
        )
        model = FEDformer.from_config({**cfg.__dict__, "moving_avg": 3, "modes": 4})
        x_enc = torch.randn(2, cfg.seq_len, cfg.enc_in)
        out = model.predict(model.make_forecast_batch(x_enc))
        self.assertEqual(out.shape, (2, cfg.pred_len, cfg.c_out))

    def test_rul_predict_api(self):
        cfg = RULConfig(
            enc_in=2,
            seq_len=32,
            d_model=32,
            n_heads=4,
            e_layers=1,
            d_ff=64,
            dropout=0.1,
        )
        model = InformerRUL.from_config(cfg)
        x = torch.randn(3, cfg.seq_len, cfg.enc_in)
        batch = model.make_rul_batch(x)
        out = model.predict(batch)
        self.assertEqual(out.shape, (3,))

    def test_mtgnn_predict_api(self):
        cfg = ForecastingConfig(
            enc_in=5,
            dec_in=5,
            c_out=5,
            seq_len=16,
            label_len=8,
            pred_len=4,
            d_model=32,
            e_layers=2,
            d_ff=64,
            dropout=0.1,
        )
        model = MTGNN.from_config(cfg)
        x_enc = torch.randn(2, cfg.seq_len, cfg.enc_in)
        batch = model.make_forecast_batch(x_enc)
        out = model.predict(batch)
        self.assertEqual(out.shape, (2, cfg.pred_len, cfg.c_out))

    def test_stemgnn_predict_api(self):
        cfg = ForecastingConfig(
            enc_in=4,
            dec_in=4,
            c_out=4,
            seq_len=12,
            label_len=6,
            pred_len=3,
            d_model=16,
            e_layers=2,
            dropout=0.1,
        )
        model = StemGNN.from_config(cfg)
        x_enc = torch.randn(2, cfg.seq_len, cfg.enc_in)
        batch = model.make_forecast_batch(x_enc)
        out = model.predict(batch)
        self.assertEqual(out.shape, (2, cfg.pred_len, cfg.c_out))

    def test_mtgnn_prior_graph_mode(self):
        cfg = ForecastingConfig(
            enc_in=4,
            dec_in=4,
            c_out=4,
            seq_len=12,
            label_len=6,
            pred_len=3,
            d_model=16,
            e_layers=2,
            dropout=0.1,
        )
        adjacency = np.eye(4, dtype=np.float32)
        model = MTGNN.from_config({**cfg.__dict__, "graph_mode": "prior", "adjacency": adjacency})
        x_enc = torch.randn(2, cfg.seq_len, cfg.enc_in)
        out = model.predict(model.make_forecast_batch(x_enc))
        self.assertEqual(out.shape, (2, cfg.pred_len, cfg.c_out))

    def test_stemgnn_hybrid_graph_mode(self):
        cfg = ForecastingConfig(
            enc_in=4,
            dec_in=4,
            c_out=4,
            seq_len=12,
            label_len=6,
            pred_len=3,
            d_model=16,
            e_layers=2,
            dropout=0.1,
        )
        adjacency = np.eye(4, dtype=np.float32)
        model = StemGNN.from_config(
            {**cfg.__dict__, "graph_mode": "hybrid", "adjacency": adjacency, "prior_weight": 0.7}
        )
        x_enc = torch.randn(2, cfg.seq_len, cfg.enc_in)
        out = model.predict(model.make_forecast_batch(x_enc))
        self.assertEqual(out.shape, (2, cfg.pred_len, cfg.c_out))

    def test_patchtst_predict_api(self):
        cfg = ForecastingConfig(
            enc_in=3,
            dec_in=3,
            c_out=3,
            seq_len=16,
            label_len=8,
            pred_len=4,
            d_model=16,
            n_heads=4,
            e_layers=2,
            d_ff=32,
            dropout=0.1,
        )
        model = PatchTST.from_config({**cfg.__dict__, "patch_len": 4})
        x_enc = torch.randn(2, cfg.seq_len, cfg.enc_in)
        out = model.predict(model.make_forecast_batch(x_enc))
        self.assertEqual(out.shape, (2, cfg.pred_len, cfg.c_out))

    def test_scinet_predict_api(self):
        cfg = ForecastingConfig(
            enc_in=3,
            dec_in=3,
            c_out=3,
            seq_len=16,
            label_len=8,
            pred_len=4,
            d_model=16,
            n_heads=4,
            e_layers=2,
            d_layers=1,
            d_ff=32,
            dropout=0.1,
        )
        model = SCINet.from_config({**cfg.__dict__, "levels": 2, "kernel_size": 3})
        x_enc = torch.randn(2, cfg.seq_len, cfg.enc_in)
        out = model.predict(model.make_forecast_batch(x_enc))
        self.assertEqual(out.shape, (2, cfg.pred_len, cfg.c_out))

    def test_ast_predict_api(self):
        cfg = ForecastingConfig(
            enc_in=3,
            dec_in=3,
            c_out=3,
            seq_len=16,
            label_len=8,
            pred_len=4,
            d_model=16,
            n_heads=4,
            e_layers=2,
            d_ff=32,
            dropout=0.1,
        )
        model = AST.from_config({**cfg.__dict__, "top_k": 4})
        x_enc = torch.randn(2, cfg.seq_len, cfg.enc_in)
        out = model.predict(model.make_forecast_batch(x_enc))
        self.assertEqual(out.shape, (2, cfg.pred_len, cfg.c_out))

    def test_itransformer_predict_api(self):
        cfg = ForecastingConfig(
            enc_in=4,
            dec_in=4,
            c_out=4,
            seq_len=12,
            label_len=6,
            pred_len=3,
            d_model=16,
            n_heads=4,
            e_layers=2,
            d_ff=32,
            dropout=0.1,
        )
        model = ITransformer.from_config(cfg)
        x_enc = torch.randn(2, cfg.seq_len, cfg.enc_in)
        out = model.predict(model.make_forecast_batch(x_enc))
        self.assertEqual(out.shape, (2, cfg.pred_len, cfg.c_out))

    def test_nbeats_predict_api(self):
        cfg = ForecastingConfig(
            enc_in=3,
            dec_in=3,
            c_out=3,
            seq_len=16,
            label_len=8,
            pred_len=4,
            d_model=32,
            e_layers=3,
            d_ff=64,
            dropout=0.1,
        )
        model = NBEATS.from_config({**cfg.__dict__, "stack_layers": 3})
        x_enc = torch.randn(2, cfg.seq_len, cfg.enc_in)
        out = model.predict(model.make_forecast_batch(x_enc))
        self.assertEqual(out.shape, (2, cfg.pred_len, cfg.c_out))

    def test_dlinear_predict_api(self):
        cfg = ForecastingConfig(
            enc_in=3,
            dec_in=3,
            c_out=3,
            seq_len=16,
            label_len=8,
            pred_len=4,
            d_model=16,
            e_layers=1,
            d_ff=32,
            dropout=0.1,
        )
        model = DLinear.from_config({**cfg.__dict__, "moving_avg": 3, "individual": True})
        x_enc = torch.randn(2, cfg.seq_len, cfg.enc_in)
        out = model.predict(model.make_forecast_batch(x_enc))
        self.assertEqual(out.shape, (2, cfg.pred_len, cfg.c_out))

    def test_dlinear_shared_head_predict_api(self):
        model = create_model(
            "DLinear",
            domain="forecasting",
            enc_in=4,
            dec_in=4,
            c_out=2,
            seq_len=12,
            label_len=6,
            pred_len=3,
            moving_avg=3,
            individual=False,
            use_norm=True,
        )
        x_enc = torch.randn(2, 12, 4)
        out = model.predict(model.make_forecast_batch(x_enc))
        self.assertEqual(out.shape, (2, 3, 2))

    def test_nhits_predict_api(self):
        cfg = ForecastingConfig(
            enc_in=3,
            dec_in=3,
            c_out=3,
            seq_len=16,
            label_len=8,
            pred_len=4,
            d_model=32,
            e_layers=3,
            d_ff=64,
            dropout=0.1,
        )
        model = NHITS.from_config(
            {
                **cfg.__dict__,
                "stack_layers": 2,
                "pooling_kernels": (1, 2),
                "downsample_frequencies": (1, 2),
            }
        )
        x_enc = torch.randn(2, cfg.seq_len, cfg.enc_in)
        out = model.predict(model.make_forecast_batch(x_enc))
        self.assertEqual(out.shape, (2, cfg.pred_len, cfg.c_out))

    def test_nhits_nearest_interpolation_predict_api(self):
        model = create_model(
            "NHITS",
            domain="forecasting",
            enc_in=4,
            dec_in=4,
            c_out=2,
            seq_len=12,
            label_len=6,
            pred_len=3,
            d_ff=32,
            e_layers=2,
            pooling_kernels=(1, 3),
            downsample_frequencies=(1, 2),
            interpolation_mode="nearest",
            use_norm=True,
        )
        x_enc = torch.randn(2, 12, 4)
        out = model.predict(model.make_forecast_batch(x_enc))
        self.assertEqual(out.shape, (2, 3, 2))

    def test_graphwavenet_predict_api(self):
        cfg = ForecastingConfig(
            enc_in=4,
            dec_in=4,
            c_out=4,
            seq_len=16,
            label_len=8,
            pred_len=4,
            d_model=32,
            e_layers=2,
            d_ff=64,
            dropout=0.1,
        )
        adjacency = np.eye(4, dtype=np.float32)
        model = GraphWaveNet.from_config(
            {**cfg.__dict__, "graph_mode": "hybrid", "adjacency": adjacency, "prior_weight": 0.6}
        )
        x_enc = torch.randn(2, cfg.seq_len, cfg.enc_in)
        out = model.predict(model.make_forecast_batch(x_enc))
        self.assertEqual(out.shape, (2, cfg.pred_len, cfg.c_out))

    def test_agcrn_predict_api(self):
        cfg = ForecastingConfig(
            enc_in=4,
            dec_in=4,
            c_out=4,
            seq_len=12,
            label_len=6,
            pred_len=3,
            d_model=16,
            e_layers=1,
            d_ff=32,
            dropout=0.1,
        )
        adjacency = np.eye(4, dtype=np.float32)
        model = AGCRN.from_config(
            {**cfg.__dict__, "graph_mode": "hybrid", "adjacency": adjacency, "prior_weight": 0.6}
        )
        x_enc = torch.randn(2, cfg.seq_len, cfg.enc_in)
        out = model.predict(model.make_forecast_batch(x_enc))
        self.assertEqual(out.shape, (2, cfg.pred_len, cfg.c_out))
        self.assertEqual(model.learned_adjacency().shape, (4, 4))

    def test_d2stgnn_predict_api(self):
        cfg = ForecastingConfig(
            enc_in=4,
            dec_in=4,
            c_out=4,
            seq_len=12,
            label_len=6,
            pred_len=3,
            d_model=16,
            e_layers=2,
            d_ff=32,
            dropout=0.1,
        )
        adjacency = np.eye(4, dtype=np.float32)
        model = D2STGNN.from_config(
            {**cfg.__dict__, "graph_mode": "hybrid", "adjacency": adjacency, "prior_weight": 0.6}
        )
        x_enc = torch.randn(2, cfg.seq_len, cfg.enc_in)
        out = model.predict(model.make_forecast_batch(x_enc))
        self.assertEqual(out.shape, (2, cfg.pred_len, cfg.c_out))
        self.assertEqual(model.learned_adjacency(x_enc).shape, (4, 4))

    def test_dcrnn_predict_api(self):
        cfg = ForecastingConfig(
            enc_in=4,
            dec_in=4,
            c_out=4,
            seq_len=12,
            label_len=6,
            pred_len=3,
            d_model=16,
            e_layers=2,
            d_ff=32,
            dropout=0.1,
        )
        adjacency = np.eye(4, dtype=np.float32)
        model = DCRNN.from_config(
            {**cfg.__dict__, "graph_mode": "prior", "adjacency": adjacency, "diffusion_steps": 2}
        )
        x_enc = torch.randn(2, cfg.seq_len, cfg.enc_in)
        out = model.predict(model.make_forecast_batch(x_enc))
        self.assertEqual(out.shape, (2, cfg.pred_len, cfg.c_out))

    def test_fcgaga_predict_api(self):
        cfg = ForecastingConfig(
            enc_in=4,
            dec_in=4,
            c_out=4,
            seq_len=12,
            label_len=6,
            pred_len=3,
            d_model=16,
            e_layers=2,
            d_ff=32,
            dropout=0.1,
        )
        adjacency = np.eye(4, dtype=np.float32)
        model = FCGAGA.from_config(
            {**cfg.__dict__, "graph_mode": "hybrid", "adjacency": adjacency, "prior_weight": 0.6}
        )
        x_enc = torch.randn(2, cfg.seq_len, cfg.enc_in)
        out = model.predict(model.make_forecast_batch(x_enc))
        self.assertEqual(out.shape, (2, cfg.pred_len, cfg.c_out))

    def test_gts_predict_api(self):
        cfg = ForecastingConfig(
            enc_in=4,
            dec_in=4,
            c_out=4,
            seq_len=12,
            label_len=6,
            pred_len=3,
            d_model=16,
            e_layers=2,
            d_ff=32,
            dropout=0.1,
        )
        adjacency = np.eye(4, dtype=np.float32)
        model = GTS.from_config(
            {**cfg.__dict__, "graph_mode": "hybrid", "adjacency": adjacency, "prior_weight": 0.6}
        )
        x_enc = torch.randn(2, cfg.seq_len, cfg.enc_in)
        out = model.predict(model.make_forecast_batch(x_enc))
        learned_adj = model.learned_adjacency()
        self.assertEqual(out.shape, (2, cfg.pred_len, cfg.c_out))
        self.assertEqual(learned_adj.shape, (4, 4))

    def test_timemixer_predict_api(self):
        cfg = ForecastingConfig(
            enc_in=3,
            dec_in=3,
            c_out=3,
            seq_len=16,
            label_len=8,
            pred_len=4,
            d_model=16,
            n_heads=4,
            e_layers=2,
            d_ff=32,
            dropout=0.1,
        )
        model = TimeMixer.from_config({**cfg.__dict__, "down_sampling_layers": 2, "moving_avg": 3})
        x_enc = torch.randn(2, cfg.seq_len, cfg.enc_in)
        out = model.predict(model.make_forecast_batch(x_enc))
        self.assertEqual(out.shape, (2, cfg.pred_len, cfg.c_out))

    def test_timesnet_predict_api(self):
        cfg = ForecastingConfig(
            enc_in=3,
            dec_in=3,
            c_out=3,
            seq_len=16,
            label_len=8,
            pred_len=4,
            d_model=16,
            e_layers=2,
            d_ff=32,
            dropout=0.1,
        )
        model = TimesNet.from_config({**cfg.__dict__, "top_k": 2, "num_kernels": 3})
        x_enc = torch.randn(2, cfg.seq_len, cfg.enc_in)
        out = model.predict(model.make_forecast_batch(x_enc))
        self.assertEqual(out.shape, (2, cfg.pred_len, cfg.c_out))

    def test_tsmixer_predict_api(self):
        cfg = ForecastingConfig(
            enc_in=3,
            dec_in=3,
            c_out=3,
            seq_len=16,
            label_len=8,
            pred_len=4,
            d_model=16,
            e_layers=2,
            d_ff=32,
            dropout=0.1,
        )
        model = TSMixer.from_config(cfg)
        x_enc = torch.randn(2, cfg.seq_len, cfg.enc_in)
        out = model.predict(model.make_forecast_batch(x_enc))
        self.assertEqual(out.shape, (2, cfg.pred_len, cfg.c_out))

    def test_stdn_predict_api(self):
        cfg = ForecastingConfig(
            enc_in=4,
            dec_in=4,
            c_out=4,
            seq_len=12,
            label_len=6,
            pred_len=3,
            d_model=16,
            n_heads=4,
            e_layers=2,
            d_ff=32,
            dropout=0.1,
        )
        model = STDN.from_config(cfg)
        x_enc = torch.randn(2, cfg.seq_len, cfg.enc_in)
        out = model.predict(model.make_forecast_batch(x_enc))
        self.assertEqual(out.shape, (2, cfg.pred_len, cfg.c_out))

    def test_crossformer_predict_api(self):
        cfg = ForecastingConfig(
            enc_in=4,
            dec_in=4,
            c_out=4,
            seq_len=18,
            label_len=9,
            pred_len=3,
            d_model=16,
            n_heads=4,
            e_layers=2,
            d_ff=32,
            dropout=0.1,
        )
        model = Crossformer.from_config({**cfg.__dict__, "seg_len": 6})
        x_enc = torch.randn(2, cfg.seq_len, cfg.enc_in)
        out = model.predict(model.make_forecast_batch(x_enc))
        self.assertEqual(out.shape, (2, cfg.pred_len, cfg.c_out))

    def test_micn_predict_api(self):
        cfg = ForecastingConfig(
            enc_in=3,
            dec_in=3,
            c_out=3,
            seq_len=16,
            label_len=8,
            pred_len=4,
            d_model=16,
            e_layers=2,
            d_ff=32,
            dropout=0.1,
        )
        model = MICN.from_config({**cfg.__dict__, "moving_avg": 3, "conv_kernel_sizes": [3, 5]})
        x_enc = torch.randn(2, cfg.seq_len, cfg.enc_in)
        out = model.predict(model.make_forecast_batch(x_enc))
        self.assertEqual(out.shape, (2, cfg.pred_len, cfg.c_out))

    def test_mlcnn_predict_api(self):
        cfg = ForecastingConfig(
            enc_in=3,
            dec_in=3,
            c_out=3,
            seq_len=16,
            label_len=8,
            pred_len=4,
            d_model=16,
            e_layers=2,
            d_ff=32,
            dropout=0.1,
        )
        model = MLCNN.from_config({**cfg.__dict__, "kernel_sizes": [3, 5]})
        x_enc = torch.randn(2, cfg.seq_len, cfg.enc_in)
        out = model.predict(model.make_forecast_batch(x_enc))
        self.assertEqual(out.shape, (2, cfg.pred_len, cfg.c_out))

    def test_convtimenet_predict_api(self):
        cfg = ForecastingConfig(
            enc_in=3,
            dec_in=3,
            c_out=3,
            seq_len=16,
            label_len=8,
            pred_len=4,
            d_model=16,
            e_layers=2,
            d_ff=32,
            dropout=0.1,
        )
        model = ConvTimeNet.from_config(
            {**cfg.__dict__, "patch_len": 4, "stride": 2, "large_kernel": 5, "multiscale_kernels": (3, 5)}
        )
        x_enc = torch.randn(2, cfg.seq_len, cfg.enc_in)
        out = model.predict(model.make_forecast_batch(x_enc))
        self.assertEqual(out.shape, (2, cfg.pred_len, cfg.c_out))

    def test_softs_predict_api(self):
        cfg = ForecastingConfig(
            enc_in=4,
            dec_in=4,
            c_out=4,
            seq_len=16,
            label_len=8,
            pred_len=4,
            d_model=16,
            e_layers=2,
            d_ff=32,
            dropout=0.1,
        )
        model = SOFTS.from_config({**cfg.__dict__, "core_dim": 8, "pooling": "attention"})
        x_enc = torch.randn(2, cfg.seq_len, cfg.enc_in)
        out = model.predict(model.make_forecast_batch(x_enc))
        self.assertEqual(out.shape, (2, cfg.pred_len, cfg.c_out))

    def test_astgcn_predict_api(self):
        cfg = ForecastingConfig(
            enc_in=4,
            dec_in=4,
            c_out=4,
            seq_len=12,
            label_len=6,
            pred_len=3,
            d_model=16,
            e_layers=2,
            d_ff=32,
            dropout=0.1,
        )
        adjacency = np.eye(4, dtype=np.float32)
        model = ASTGCN.from_config(
            {**cfg.__dict__, "graph_mode": "hybrid", "adjacency": adjacency, "prior_weight": 0.6}
        )
        x_enc = torch.randn(2, cfg.seq_len, cfg.enc_in)
        out = model.predict(model.make_forecast_batch(x_enc))
        self.assertEqual(out.shape, (2, cfg.pred_len, cfg.c_out))

    def test_dmstgcn_predict_api(self):
        cfg = ForecastingConfig(
            enc_in=4,
            dec_in=4,
            c_out=4,
            seq_len=12,
            label_len=6,
            pred_len=3,
            d_model=16,
            e_layers=2,
            d_ff=32,
            dropout=0.1,
        )
        adjacency = np.eye(4, dtype=np.float32)
        model = DMSTGCN.from_config(
            {**cfg.__dict__, "graph_mode": "hybrid", "adjacency": adjacency, "prior_weight": 0.6}
        )
        x_enc = torch.randn(2, cfg.seq_len, cfg.enc_in)
        out = model.predict(model.make_forecast_batch(x_enc))
        self.assertEqual(out.shape, (2, cfg.pred_len, cfg.c_out))

    def test_stfgnn_predict_api(self):
        cfg = ForecastingConfig(
            enc_in=4,
            dec_in=4,
            c_out=4,
            seq_len=12,
            label_len=6,
            pred_len=3,
            d_model=16,
            e_layers=2,
            d_ff=32,
            dropout=0.1,
        )
        adjacency = np.eye(4, dtype=np.float32)
        model = STFGNN.from_config(
            {**cfg.__dict__, "graph_mode": "hybrid", "adjacency": adjacency, "prior_weight": 0.6}
        )
        x_enc = torch.randn(2, cfg.seq_len, cfg.enc_in)
        out = model.predict(model.make_forecast_batch(x_enc))
        self.assertEqual(out.shape, (2, cfg.pred_len, cfg.c_out))

    def test_stgode_predict_api(self):
        cfg = ForecastingConfig(
            enc_in=4,
            dec_in=4,
            c_out=4,
            seq_len=12,
            label_len=6,
            pred_len=3,
            d_model=16,
            e_layers=2,
            d_ff=32,
            dropout=0.1,
        )
        adjacency = np.eye(4, dtype=np.float32)
        semantic = np.full((4, 4), 0.25, dtype=np.float32)
        model = STGODE.from_config(
            {
                **cfg.__dict__,
                "graph_mode": "hybrid",
                "adjacency": adjacency,
                "semantic_adjacency": semantic,
                "prior_weight": 0.6,
                "ode_steps": 2,
            }
        )
        x_enc = torch.randn(2, cfg.seq_len, cfg.enc_in)
        out = model.predict(model.make_forecast_batch(x_enc))
        self.assertEqual(out.shape, (2, cfg.pred_len, cfg.c_out))

    def test_thgnn_predict_api(self):
        model = THGNN(enc_in=4, seq_len=32, d_model=16, dropout=0.1)
        x = torch.randn(3, 32, 4)
        out = model.predict(model.make_rul_batch(x))
        self.assertEqual(out.shape, (3,))

    def test_timexer_predict_api(self):
        model = TimeXer(
            enc_in=4,
            dec_in=2,
            c_out=2,
            seq_len=16,
            label_len=8,
            pred_len=4,
            d_model=16,
            n_heads=4,
            e_layers=2,
            d_ff=32,
            patch_len=4,
            dropout=0.1,
        )
        x_enc = torch.randn(2, 16, 4)
        x_target_enc = x_enc[..., :2]
        x_exog_enc = x_enc[..., 2:]
        x_target_dec = torch.randn(2, 12, 2)
        batch = model.make_forecast_batch(
            x_enc=x_enc,
            x_target_enc=x_target_enc,
            x_exog_enc=x_exog_enc,
            x_target_dec=x_target_dec,
            x_exog_dec=torch.randn(2, 12, 2),
        )
        out = model.predict(batch)
        self.assertEqual(out.shape, (2, 4, 2))

    def test_tft_predict_api(self):
        cfg = ForecastingConfig(
            enc_in=4,
            dec_in=4,
            c_out=4,
            seq_len=16,
            label_len=8,
            pred_len=4,
            d_model=16,
            n_heads=4,
            e_layers=1,
            d_ff=32,
            dropout=0.1,
        )
        model = TFT.from_config({**cfg.__dict__, "activation": "elu"})
        x_enc = torch.randn(2, cfg.seq_len, cfg.enc_in)
        out = model.predict(model.make_forecast_batch(x_enc))
        self.assertEqual(out.shape, (2, cfg.pred_len, cfg.c_out))
        self.assertIsNotNone(model.attention_weights())

    def test_tft_exogenous_predict_api(self):
        model = TFT(
            enc_in=8,
            dec_in=6,
            c_out=2,
            seq_len=16,
            label_len=8,
            pred_len=4,
            d_model=16,
            n_heads=4,
            e_layers=1,
            d_ff=32,
            dropout=0.1,
        )
        x_enc = torch.randn(2, 16, 4)
        batch = model.make_forecast_batch(
            x_enc=x_enc,
            x_target_enc=x_enc[..., :2],
            x_exog_enc=x_enc[..., 2:],
            x_target_dec=torch.randn(2, 12, 2),
            x_exog_dec=torch.randn(2, 12, 2),
        )
        out = model.predict(batch)
        self.assertEqual(out.shape, (2, 4, 2))

    def test_dag_exogenous_predict_api(self):
        model = DAG(
            enc_in=4,
            dec_in=2,
            c_out=2,
            seq_len=16,
            label_len=8,
            pred_len=4,
            d_model=16,
            n_heads=4,
            e_layers=2,
            d_ff=32,
            dropout=0.1,
        )
        x_enc = torch.randn(2, 16, 4)
        batch = model.make_forecast_batch(
            x_enc=x_enc,
            x_target_enc=x_enc[..., :2],
            x_exog_enc=x_enc[..., 2:],
            x_target_dec=torch.randn(2, 12, 2),
            x_exog_dec=torch.randn(2, 12, 2),
        )
        out = model.predict(batch)
        self.assertEqual(out.shape, (2, 4, 2))

    def test_tide_exogenous_predict_api(self):
        model = TiDE(
            enc_in=4,
            dec_in=2,
            c_out=2,
            seq_len=16,
            label_len=8,
            pred_len=4,
            d_model=16,
            e_layers=2,
            d_ff=32,
            dropout=0.1,
            temporal_decoder_dim=8,
        )
        x_enc = torch.randn(2, 16, 4)
        batch = model.make_forecast_batch(
            x_enc=x_enc,
            x_target_enc=x_enc[..., :2],
            x_exog_enc=x_enc[..., 2:],
            x_target_dec=torch.randn(2, 12, 2),
            x_exog_dec=torch.randn(2, 12, 2),
        )
        out = model.predict(batch)
        self.assertEqual(out.shape, (2, 4, 2))

    def test_koopa_predict_api(self):
        cfg = ForecastingConfig(
            enc_in=3,
            dec_in=3,
            c_out=3,
            seq_len=16,
            label_len=8,
            pred_len=4,
            d_model=16,
            e_layers=2,
            d_ff=32,
            dropout=0.1,
        )
        model = Koopa.from_config({**cfg.__dict__, "modes": 4, "koopman_dim": 16, "num_predictors": 2})
        x_enc = torch.randn(2, cfg.seq_len, cfg.enc_in)
        out = model.predict(model.make_forecast_batch(x_enc))
        self.assertEqual(out.shape, (2, cfg.pred_len, cfg.c_out))

    def test_frets_predict_api(self):
        cfg = ForecastingConfig(
            enc_in=4,
            dec_in=4,
            c_out=4,
            seq_len=16,
            label_len=8,
            pred_len=4,
            d_model=16,
            e_layers=2,
            d_ff=32,
            dropout=0.1,
        )
        model = FreTS.from_config({**cfg.__dict__, "sparsity_threshold": 0.01})
        x_enc = torch.randn(2, cfg.seq_len, cfg.enc_in)
        out = model.predict(model.make_forecast_batch(x_enc))
        self.assertEqual(out.shape, (2, cfg.pred_len, cfg.c_out))

    def test_deepar_probabilistic_api(self):
        model = DeepAR(
            enc_in=3,
            dec_in=3,
            c_out=3,
            seq_len=12,
            label_len=6,
            pred_len=4,
            d_model=16,
            e_layers=2,
            dropout=0.1,
            num_samples=7,
        )
        x_enc = torch.randn(2, 12, 3)
        y_true = torch.randn(2, 4, 3)
        batch = model.make_probabilistic_forecast_batch(x_enc=x_enc, y_true=y_true)
        mean = model.predict(batch)
        dist = model.predict_distribution(batch)
        samples = model.sample(batch)
        loss = model.fit_step(batch)
        self.assertEqual(mean.shape, (2, 4, 3))
        self.assertEqual(dist["mean"].shape, (2, 4, 3))
        self.assertEqual(dist["scale"].shape, (2, 4, 3))
        self.assertEqual(samples.shape, (7, 2, 4, 3))
        self.assertEqual(loss.dim(), 0)

    def test_timegrad_probabilistic_api(self):
        model = TimeGrad(
            enc_in=3,
            dec_in=3,
            c_out=3,
            seq_len=12,
            label_len=6,
            pred_len=4,
            d_model=16,
            e_layers=2,
            dropout=0.1,
            num_samples=5,
            diffusion_steps=4,
        )
        x_enc = torch.randn(2, 12, 3)
        y_true = torch.randn(2, 4, 3)
        batch = model.make_probabilistic_forecast_batch(x_enc=x_enc, y_true=y_true)
        mean = model.predict(batch)
        dist = model.predict_distribution(batch)
        samples = model.sample(batch)
        loss = model.fit_step(batch)
        self.assertEqual(mean.shape, (2, 4, 3))
        self.assertEqual(dist["mean"].shape, (2, 4, 3))
        self.assertEqual(dist["scale"].shape, (2, 4, 3))
        self.assertEqual(samples.shape, (5, 2, 4, 3))
        self.assertEqual(loss.dim(), 0)

    def test_protran_probabilistic_api(self):
        model = ProTran(
            enc_in=3,
            dec_in=3,
            c_out=3,
            seq_len=12,
            label_len=6,
            pred_len=4,
            d_model=16,
            n_heads=4,
            e_layers=2,
            d_ff=32,
            dropout=0.1,
            num_samples=6,
        )
        x_enc = torch.randn(2, 12, 3)
        y_true = torch.randn(2, 4, 3)
        batch = model.make_probabilistic_forecast_batch(x_enc=x_enc, y_true=y_true)
        mean = model.predict(batch)
        dist = model.predict_distribution(batch)
        samples = model.sample(batch)
        loss = model.fit_step(batch)
        self.assertEqual(mean.shape, (2, 4, 3))
        self.assertEqual(dist["mean"].shape, (2, 4, 3))
        self.assertEqual(dist["scale"].shape, (2, 4, 3))
        self.assertEqual(samples.shape, (6, 2, 4, 3))
        self.assertEqual(loss.dim(), 0)

    def test_ppm_probabilistic_api(self):
        model = PPM(
            enc_in=3,
            dec_in=3,
            c_out=3,
            seq_len=12,
            label_len=6,
            pred_len=4,
            d_model=16,
            n_heads=4,
            e_layers=2,
            d_ff=32,
            dropout=0.1,
            num_samples=6,
        )
        x_enc = torch.randn(2, 12, 3)
        y_true = torch.randn(2, 4, 3)
        batch = model.make_probabilistic_forecast_batch(x_enc=x_enc, y_true=y_true)
        mean = model.predict(batch)
        dist = model.predict_distribution(batch)
        samples = model.sample(batch)
        loss = model.fit_step(batch)
        self.assertEqual(mean.shape, (2, 4, 3))
        self.assertEqual(dist["mean"].shape, (2, 4, 3))
        self.assertEqual(dist["scale"].shape, (2, 4, 3))
        self.assertEqual(dist["prior_hidden"].shape[:2], (2, 4))
        self.assertEqual(samples.shape, (6, 2, 4, 3))
        self.assertEqual(loss.dim(), 0)


if __name__ == "__main__":
    unittest.main()
