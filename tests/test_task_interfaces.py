import unittest

import torch
import torch.nn.functional as F

import torch_like as tl


class TaskInterfaceTests(unittest.TestCase):
    def test_graph_pooling_excludes_masked_nodes_and_expands_graph_ids(self):
        model = tl.GCN(
            input_dim=2,
            output_dim=1,
            num_nodes=3,
            d_model=4,
            num_layers=1,
            dropout=0.0,
            task_level="graph",
        ).eval()
        x = torch.randn(2, 3, 2)
        adjacency = torch.eye(3)
        node_mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.float32)

        graph_out = model(x, adjacency=adjacency, node_mask=node_mask, batch_index=torch.tensor([0, 1]))
        model.task_level = "node"
        node_out = model(x, adjacency=adjacency, node_mask=node_mask)
        expected = node_out.sum(dim=1) / node_mask.sum(dim=1, keepdim=True)
        self.assertTrue(torch.allclose(graph_out, expected))

    def test_graph_pooling_rejects_invalid_batch_index(self):
        model = tl.GCN(input_dim=2, output_dim=1, num_nodes=3, task_level="graph")
        with self.assertRaisesRegex(ValueError, "batch_index must contain"):
            model(torch.randn(2, 3, 2), adjacency=torch.eye(3), batch_index=torch.tensor([0, 1, 2]))

    def test_causal_lm_loss_uses_next_token_targets(self):
        model = tl.TransformerDecoder(vocab_size=5, max_seq_len=4, d_model=8, n_heads=2, n_layers=1, d_ff=16)
        labels = torch.tensor([[1, 2, 3]])
        logits = torch.full((1, 3, 5), -10.0)
        logits[0, 0, 2] = 10.0
        logits[0, 1, 3] = 10.0
        loss = model.causal_lm_loss(logits, labels)
        same_position_loss = F.cross_entropy(logits.reshape(-1, 5), labels.reshape(-1))
        self.assertLess(loss.item(), 1e-4)
        self.assertGreater(same_position_loss.item(), 1.0)

    def test_diagnosis_batch_helper(self):
        class DummyDiagnosis(tl.DiagnosisModel):
            def __init__(self):
                super().__init__()
                self._config = tl.DiagnosisConfig(input_dim=2, num_classes=3, seq_len=8)

            def forward(self, x, x_mark=None):
                return x.mean(dim=1)

        model = DummyDiagnosis()
        batch = model.make_diagnosis_batch(torch.randn(4, 8, 2))
        out = model.predict(batch)
        self.assertEqual(out.shape, (4, 2))

    def test_detection_batch_helper(self):
        class DummyDetection(tl.DetectionModel):
            def __init__(self):
                super().__init__()
                self._config = tl.DetectionConfig(input_dim=2, seq_len=8)

            def forward(self, x, x_mark=None):
                return x.max(dim=1).values

        model = DummyDetection()
        batch = model.make_detection_batch(torch.randn(4, 8, 2))
        out = model.predict(batch)
        self.assertEqual(out.shape, (4, 2))

    def test_graph_batch_helper(self):
        class DummyGraph(tl.GraphModel):
            def __init__(self):
                super().__init__()
                self._config = tl.GraphConfig(input_dim=3, output_dim=2, num_nodes=5)

            def forward(self, x, adjacency=None, edge_index=None, edge_attr=None, batch_index=None, node_mask=None):
                del adjacency, edge_index, edge_attr, batch_index, node_mask
                return x[..., :2]

        model = DummyGraph()
        batch = model.make_graph_batch(torch.randn(4, 5, 3), adjacency=torch.eye(5).expand(4, 5, 5))
        out = model.predict(batch)
        self.assertEqual(out.shape, (4, 5, 2))

    def test_probabilistic_forecasting_batch_helper(self):
        class DummyProbForecast(tl.ProbabilisticForecastingModel):
            def __init__(self):
                super().__init__()
                self.label_len = 4
                self.pred_len = 2
                self._config = tl.ProbabilisticForecastingConfig(
                    enc_in=3, dec_in=3, c_out=3, seq_len=8, label_len=4, pred_len=2
                )

            def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
                del x_mark_enc, x_dec, x_mark_dec, enc_self_mask, dec_self_mask, dec_enc_mask
                return x_enc[:, -self.pred_len :, :]

            def predict_distribution(self, batch):
                mean = self.predict(batch)
                scale = torch.ones_like(mean)
                return {"mean": mean, "scale": scale}

            def sample(self, batch, num_samples: int = 100):
                mean = self.predict(batch)
                return mean.unsqueeze(0).expand(num_samples, -1, -1, -1)

            def fit_step(self, batch):
                if batch.y_true is None:
                    raise ValueError("missing y_true")
                mean = self.predict(batch)
                return torch.mean((mean - batch.y_true) ** 2)

        model = DummyProbForecast()
        x = torch.randn(4, 8, 3)
        y = torch.randn(4, 2, 3)
        batch = model.make_probabilistic_forecast_batch(x_enc=x, y_true=y)
        out = model.predict(batch)
        dist = model.predict_distribution(batch)
        samples = model.sample(batch, num_samples=5)
        loss = model.fit_step(batch)
        self.assertEqual(out.shape, (4, 2, 3))
        self.assertEqual(dist["mean"].shape, (4, 2, 3))
        self.assertEqual(samples.shape, (5, 4, 2, 3))
        self.assertEqual(loss.dim(), 0)

    def test_sequence_batch_helper(self):
        class DummySequence(tl.SequenceModel):
            def __init__(self):
                super().__init__()
                self._config = tl.SequenceConfig(vocab_size=32, max_seq_len=8, d_model=16, n_heads=2, n_layers=2, d_ff=32)

            def forward(self, input_ids, attention_mask=None, position_ids=None, causal_mask=None):
                del attention_mask, position_ids, causal_mask
                batch, seq_len = input_ids.shape
                return torch.zeros(batch, seq_len, 32, device=input_ids.device)

            def loss_step(self, batch):
                if batch.labels is None:
                    raise ValueError("missing labels")
                logits = self.next_token_logits(batch)
                return logits.mean() + batch.labels.float().mean() * 0.0

        model = DummySequence()
        input_ids = torch.randint(0, 32, (4, 8))
        labels = torch.randint(0, 32, (4, 8))
        batch = model.make_sequence_batch(input_ids=input_ids, labels=labels)
        logits = model.next_token_logits(batch)
        loss = model.loss_step(batch)
        self.assertEqual(logits.shape, (4, 8, 32))
        self.assertEqual(batch.attention_mask.shape, (4, 8))
        self.assertEqual(batch.position_ids.shape, (4, 8))
        self.assertEqual(batch.causal_mask.shape, (8, 8))
        self.assertEqual(loss.dim(), 0)

    def test_transformer_decoder_sequence_api(self):
        model = tl.TransformerDecoder(vocab_size=32, max_seq_len=12, d_model=16, n_heads=4, n_layers=2, d_ff=32)
        input_ids = torch.randint(0, 32, (3, 8))
        labels = torch.randint(0, 32, (3, 8))
        batch = model.make_sequence_batch(input_ids=input_ids, labels=labels)
        logits = model.next_token_logits(batch)
        loss = model.loss_step(batch)
        generated = model.generate(model.make_sequence_batch(input_ids=input_ids[:, :4]), max_new_tokens=3)
        self.assertEqual(logits.shape, (3, 8, 32))
        self.assertEqual(loss.dim(), 0)
        self.assertEqual(generated.shape, (3, 7))

    def test_gpt_sequence_api(self):
        model = tl.GPT(vocab_size=32, max_seq_len=12, d_model=16, n_heads=4, n_layers=2, d_ff=32)
        input_ids = torch.randint(0, 32, (3, 8))
        labels = torch.randint(0, 32, (3, 8))
        labels[:, 1::2] = -100
        batch = model.make_sequence_batch(input_ids=input_ids, labels=labels)
        logits = model.next_token_logits(batch)
        features = model.forward_features(batch.input_ids, attention_mask=batch.attention_mask)
        loss = model.loss_step(batch)
        generated = model.generate(model.make_sequence_batch(input_ids=input_ids[:, :4]), max_new_tokens=3)
        self.assertEqual(logits.shape, (3, 8, 32))
        self.assertEqual(features.shape, (3, 8, 16))
        self.assertEqual(loss.dim(), 0)
        self.assertEqual(generated.shape, (3, 7))

    def test_mamba_like_sequence_api(self):
        model = tl.MambaLike(vocab_size=32, max_seq_len=12, d_model=16, n_layers=2, d_ff=32)
        input_ids = torch.randint(0, 32, (3, 8))
        labels = torch.randint(0, 32, (3, 8))
        batch = model.make_sequence_batch(input_ids=input_ids, labels=labels)
        logits = model.next_token_logits(batch)
        loss = model.loss_step(batch)
        generated = model.generate(model.make_sequence_batch(input_ids=input_ids[:, :4]), max_new_tokens=3)
        self.assertEqual(logits.shape, (3, 8, 32))
        self.assertEqual(loss.dim(), 0)
        self.assertEqual(generated.shape, (3, 7))

    def test_mamba_sequence_api(self):
        model = tl.Mamba(vocab_size=32, max_seq_len=12, d_model=16, n_layers=2, d_ff=32, d_state=8, d_conv=3, expand=2)
        input_ids = torch.randint(0, 32, (3, 8))
        labels = torch.randint(0, 32, (3, 8))
        batch = model.make_sequence_batch(input_ids=input_ids, labels=labels)
        logits = model.next_token_logits(batch)
        loss = model.loss_step(batch)
        generated = model.generate(model.make_sequence_batch(input_ids=input_ids[:, :4]), max_new_tokens=3)
        self.assertEqual(logits.shape, (3, 8, 32))
        self.assertEqual(loss.dim(), 0)
        self.assertEqual(generated.shape, (3, 7))

    def test_lru_sequence_api(self):
        model = tl.LRU(
            vocab_size=32,
            max_seq_len=12,
            d_model=16,
            n_layers=2,
            d_ff=32,
            d_state=8,
            gate_mode="none",
            skip_mode="learnable",
            radius_min=0.7,
            radius_max=0.95,
            phase_max=1.0,
        )
        input_ids = torch.randint(0, 32, (3, 8))
        labels = torch.randint(0, 32, (3, 8))
        batch = model.make_sequence_batch(input_ids=input_ids, labels=labels)
        logits = model.next_token_logits(batch)
        loss = model.loss_step(batch)
        generated = model.generate(model.make_sequence_batch(input_ids=input_ids[:, :4]), max_new_tokens=3)
        self.assertEqual(logits.shape, (3, 8, 32))
        self.assertEqual(loss.dim(), 0)
        self.assertEqual(generated.shape, (3, 7))

    def test_lru_rejects_invalid_modes(self):
        with self.assertRaises(ValueError):
            tl.LRU(vocab_size=32, max_seq_len=12, gate_mode="invalid")
        with self.assertRaises(ValueError):
            tl.LRU(vocab_size=32, max_seq_len=12, skip_mode="invalid")
        with self.assertRaises(ValueError):
            tl.LRU(vocab_size=32, max_seq_len=12, radius_min=0.99, radius_max=0.9)

    def test_caracal_sequence_api(self):
        model = tl.Caracal(
            vocab_size=32,
            max_seq_len=12,
            d_model=16,
            n_heads=4,
            n_layers=3,
            d_ff=32,
            attention_interval=2,
            window_size=4,
        )
        input_ids = torch.randint(0, 32, (3, 8))
        labels = torch.randint(0, 32, (3, 8))
        batch = model.make_sequence_batch(input_ids=input_ids, labels=labels)
        logits = model.next_token_logits(batch)
        loss = model.loss_step(batch)
        generated = model.generate(model.make_sequence_batch(input_ids=input_ids[:, :4]), max_new_tokens=3)
        self.assertEqual(logits.shape, (3, 8, 32))
        self.assertEqual(loss.dim(), 0)
        self.assertEqual(generated.shape, (3, 7))

    def test_bert_sequence_api(self):
        model = tl.BERT(vocab_size=32, max_seq_len=12, d_model=16, n_heads=4, n_layers=2, d_ff=32)
        input_ids = torch.randint(0, 32, (3, 8))
        labels = torch.randint(0, 32, (3, 8))
        labels[:, 1::2] = -100
        batch = model.make_sequence_batch(input_ids=input_ids, labels=labels)
        logits = model.next_token_logits(batch)
        loss = model.loss_step(batch)
        pooled = model.pooled_output(batch.input_ids, attention_mask=batch.attention_mask, position_ids=batch.position_ids)
        self.assertEqual(logits.shape, (3, 8, 32))
        self.assertEqual(loss.dim(), 0)
        self.assertEqual(pooled.shape, (3, 16))
        with self.assertRaises(NotImplementedError):
            model.generate(batch, max_new_tokens=2)

    def test_t5_seq2seq_api(self):
        model = tl.T5(vocab_size=32, max_seq_len=12, d_model=16, n_heads=4, n_layers=2, d_ff=32)
        input_ids = torch.randint(1, 32, (3, 7))
        decoder_input_ids = torch.randint(1, 32, (3, 5))
        labels = torch.randint(0, 32, (3, 5))
        batch = model.make_seq2seq_batch(input_ids=input_ids, decoder_input_ids=decoder_input_ids, labels=labels)
        logits = model.predict(batch)
        loss = model.loss_step(batch)
        generated = model.generate(input_ids, max_new_tokens=3)
        self.assertEqual(logits.shape, (3, 5, 32))
        self.assertEqual(batch.attention_mask.shape, (3, 7))
        self.assertEqual(batch.decoder_attention_mask.shape, (3, 5))
        self.assertEqual(loss.dim(), 0)
        self.assertEqual(generated.shape, (3, 4))

    def test_gcn_predict_api(self):
        model = tl.GCN(input_dim=3, output_dim=2, num_nodes=5, d_model=8, num_layers=2)
        batch = model.make_graph_batch(torch.randn(4, 5, 3), adjacency=torch.eye(5).expand(4, 5, 5))
        out = model.predict(batch)
        self.assertEqual(out.shape, (4, 5, 2))

    def test_gat_predict_api(self):
        model = tl.GAT(input_dim=3, output_dim=2, num_nodes=5, d_model=8, num_layers=2, heads=2)
        batch = model.make_graph_batch(torch.randn(4, 5, 3), adjacency=torch.eye(5).expand(4, 5, 5))
        out = model.predict(batch)
        self.assertEqual(out.shape, (4, 5, 2))

    def test_gnan_predict_and_explain_api(self):
        model = tl.GNAN(input_dim=3, output_dim=2, num_nodes=5, d_model=8, num_layers=2, d_ff=16, num_bases=4)
        batch = model.make_graph_batch(torch.randn(4, 5, 3), adjacency=torch.eye(5).expand(4, 5, 5))
        out = model.predict(batch)
        explanation = model.explain(batch.x, batch.adjacency)
        self.assertEqual(out.shape, (4, 5, 2))
        self.assertEqual(explanation["distance_weights"].shape, (4, 2, 5, 5))
        self.assertEqual(explanation["normalized_distance"].shape, (4, 5, 5))

    def test_graph_transformer_predict_api(self):
        model = tl.GraphTransformer(input_dim=3, output_dim=2, num_nodes=5, d_model=8, num_layers=2, n_heads=2)
        batch = model.make_graph_batch(torch.randn(4, 5, 3), adjacency=torch.eye(5).expand(4, 5, 5))
        out = model.predict(batch)
        self.assertEqual(out.shape, (4, 5, 2))

    def test_graph_transformer_uses_batched_adjacency(self):
        torch.manual_seed(0)
        model = tl.GraphTransformer(
            input_dim=3, output_dim=2, num_nodes=4, d_model=8, num_layers=1, n_heads=2, dropout=0.0
        ).eval()
        x = torch.randn(1, 4, 3).expand(2, -1, -1).clone()
        adjacency = torch.stack([torch.eye(4), torch.ones(4, 4)])
        out = model.predict(model.make_graph_batch(x, adjacency=adjacency))
        self.assertGreater((out[0] - out[1]).abs().max().item(), 1e-6)

    def test_spike_graphormer_predict_api(self):
        model = tl.SpikeGraphormer(input_dim=3, output_dim=2, num_nodes=5, d_model=8, num_layers=2, n_heads=2)
        batch = model.make_graph_batch(torch.randn(4, 5, 3), adjacency=torch.eye(5).expand(4, 5, 5))
        out = model.predict(batch)
        self.assertEqual(out.shape, (4, 5, 2))

    def test_spike_graphormer_attention_projections_receive_gradients(self):
        model = tl.SpikeGraphormer(input_dim=3, output_dim=2, num_nodes=5, d_model=8, num_layers=1, n_heads=2)
        batch = model.make_graph_batch(torch.randn(2, 5, 3), adjacency=torch.ones(2, 5, 5))
        model.predict(batch).sum().backward()
        attention = model.transformer_blocks[0].attn
        self.assertIsNotNone(attention.q_proj.weight.grad)
        self.assertIsNotNone(attention.k_proj.weight.grad)
        self.assertGreater(attention.q_proj.weight.grad.norm().item(), 0.0)

    def test_graphsage_predict_api(self):
        model = tl.GraphSAGE(input_dim=3, output_dim=2, num_nodes=5, d_model=8, num_layers=2)
        batch = model.make_graph_batch(torch.randn(4, 5, 3), adjacency=torch.eye(5).expand(4, 5, 5))
        out = model.predict(batch)
        self.assertEqual(out.shape, (4, 5, 2))

    def test_gin_predict_api(self):
        model = tl.GIN(input_dim=3, output_dim=2, num_nodes=5, d_model=8, num_layers=2)
        batch = model.make_graph_batch(torch.randn(4, 5, 3), adjacency=torch.eye(5).expand(4, 5, 5))
        out = model.predict(batch)
        self.assertEqual(out.shape, (4, 5, 2))


if __name__ == "__main__":
    unittest.main()
