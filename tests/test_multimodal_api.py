import unittest

import torch

import torch_like as tl
from torch_like.models import create_model


class MultimodalApiTests(unittest.TestCase):
    def test_clip_predict_api(self):
        model = tl.CLIP(
            vocab_size=32,
            image_size=32,
            patch_size=8,
            max_text_len=8,
            d_model=16,
            embed_dim=12,
            vision_layers=1,
            vision_heads=4,
            text_layers=1,
            text_heads=4,
            d_ff=32,
        )
        image = torch.randn(3, 3, 32, 32)
        input_ids = torch.randint(1, 32, (3, 6))
        batch = model.make_multimodal_batch(image, input_ids)
        out = model.predict(batch)
        loss = model.loss_step(batch)
        self.assertEqual(out["image_embeds"].shape, (3, 12))
        self.assertEqual(out["text_embeds"].shape, (3, 12))
        self.assertEqual(out["logits_per_image"].shape, (3, 3))
        self.assertEqual(out["logits_per_text"].shape, (3, 3))
        self.assertEqual(loss.dim(), 0)

    def test_clip_from_registry(self):
        model = create_model(
            "CLIP",
            domain="multimodal",
            vocab_size=16,
            image_size=32,
            patch_size=8,
            max_text_len=8,
            d_model=16,
            embed_dim=8,
            vision_layers=1,
            vision_heads=4,
            text_layers=1,
            text_heads=4,
            d_ff=32,
        )
        image = torch.randn(2, 3, 32, 32)
        input_ids = torch.randint(1, 16, (2, 5))
        image_features = model.encode_image(image)
        text_features = model.encode_text(input_ids)
        out = model(image, input_ids)
        self.assertEqual(image_features.shape, (2, 8))
        self.assertEqual(text_features.shape, (2, 8))
        self.assertEqual(out["logits_per_image"].shape, (2, 2))


if __name__ == "__main__":
    unittest.main()
