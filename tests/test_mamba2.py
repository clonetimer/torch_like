import unittest

import torch

import torch_like as tl
from torch_like.models.sequence.mamba2 import SSD, Mamba2, SSDState


class Mamba2Tests(unittest.TestCase):
    def test_ssd_parallel_chunkwise_and_recurrent_are_equivalent(self):
        torch.manual_seed(7)
        layer = SSD(
            d_model=8,
            d_state=3,
            d_conv=3,
            expand=2,
            head_dim=4,
            n_groups=2,
            chunk_size=3,
            dropout=0.0,
        ).eval()
        x = torch.randn(2, 7, 8)

        parallel, parallel_state = layer(x, return_state=True)
        chunkwise, chunk_state = layer(x, mode="chunkwise", return_state=True)
        recurrent_parts = []
        recurrent_state = None
        for index in range(x.size(1)):
            output, recurrent_state = layer.forward_step(x[:, index], recurrent_state)
            recurrent_parts.append(output)
        recurrent = torch.stack(recurrent_parts, dim=1)

        self.assertTrue(torch.allclose(parallel, chunkwise, atol=1e-6, rtol=1e-5))
        self.assertTrue(torch.allclose(parallel, recurrent, atol=1e-6, rtol=1e-5))
        self.assertTrue(torch.allclose(parallel_state.ssm, chunk_state.ssm, atol=1e-6))
        self.assertTrue(torch.allclose(parallel_state.ssm, recurrent_state.ssm, atol=1e-6))
        self.assertTrue(torch.allclose(parallel_state.convolution, recurrent_state.convolution, atol=1e-6))

    def test_ssd_masked_tokens_do_not_change_state(self):
        torch.manual_seed(11)
        layer = SSD(d_model=8, d_state=3, d_conv=3, expand=2, head_dim=4, dropout=0.0).eval()
        x = torch.randn(2, 5, 8)
        mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
        output, state = layer(x, attention_mask=mask, return_state=True)
        prefix, prefix_state = layer(x[:1, :3], return_state=True)

        self.assertTrue(torch.allclose(output[0, :3], prefix[0], atol=1e-6))
        self.assertTrue(torch.equal(output[0, 3:], torch.zeros_like(output[0, 3:])))
        self.assertTrue(torch.allclose(state.ssm[0], prefix_state.ssm[0], atol=1e-6))
        self.assertTrue(torch.allclose(state.convolution[0], prefix_state.convolution[0], atol=1e-6))

    def test_mamba2_framework_api_and_model_modes(self):
        torch.manual_seed(13)
        model = Mamba2(
            vocab_size=31,
            max_seq_len=16,
            d_model=8,
            n_layers=2,
            d_ff=16,
            d_state=3,
            d_conv=3,
            expand=2,
            head_dim=4,
            chunk_size=2,
            dropout=0.0,
        ).eval()
        input_ids = torch.randint(0, 31, (2, 6))
        batch = model.make_sequence_batch(input_ids, labels=input_ids.clone())
        logits = model.next_token_logits(batch)
        chunkwise = model.forward_chunkwise(input_ids, batch.attention_mask)
        recurrent, states = model.forward_recurrent(input_ids)
        loss = model.loss_step(batch)
        generated = model.generate(model.make_sequence_batch(input_ids[:, :3]), max_new_tokens=2)

        self.assertEqual(logits.shape, (2, 6, 31))
        self.assertTrue(torch.allclose(logits, chunkwise, atol=1e-6, rtol=1e-5))
        self.assertTrue(torch.allclose(logits, recurrent, atol=1e-6, rtol=1e-5))
        self.assertEqual(loss.dim(), 0)
        self.assertEqual(generated.shape, (2, 5))
        self.assertEqual(len(states), 2)
        self.assertTrue(all(isinstance(state, SSDState) for state in states))

    def test_registry_and_top_level_api(self):
        self.assertIs(tl.Mamba2, Mamba2)
        info = tl.get_model_info("Mamba2", domain="sequence")
        self.assertEqual(info["integration_level"], "paper_aligned_portable_scan")
        self.assertIn("state_space_duality", info["supports"])
        created = tl.create_model(
            "Mamba2",
            domain="sequence",
            vocab_size=16,
            max_seq_len=8,
            d_model=8,
            n_layers=1,
            d_ff=16,
            d_state=2,
            expand=1,
            head_dim=4,
        )
        self.assertIsInstance(created, Mamba2)

    def test_invalid_configuration_and_state_are_rejected(self):
        with self.assertRaises(ValueError):
            SSD(d_model=7, expand=1, head_dim=4)
        layer = SSD(d_model=8, d_state=2, expand=1, head_dim=4)
        bad_state = SSDState(torch.zeros(1, 1, 1, 1), torch.zeros(1, 1, 1))
        with self.assertRaises(ValueError):
            layer(torch.randn(1, 2, 8), state=bad_state)


if __name__ == "__main__":
    unittest.main()
