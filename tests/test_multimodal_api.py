import unittest

import torch

import torch_like as tl
from torch_like.models import create_model


class MultimodalApiTests(unittest.TestCase):
    def test_siglip_pairwise_sigmoid_api(self):
        model = tl.SigLIP(
            vocab_size=32,
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
            logit_bias_init=0.0,
        )
        batch = model.make_multimodal_batch(torch.randn(3, 3, 32, 32), torch.randint(1, 32, (3, 6)))
        output = model.predict(batch)
        loss = model.loss_step(batch)
        loss.backward()
        self.assertEqual(output["logits_per_image"].shape, (3, 3))
        self.assertEqual(loss.dim(), 0)
        self.assertIsNotNone(model.logit_bias.grad)

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

    def test_blip_vision_language_generation_api(self):
        model = tl.BLIP(
            vocab_size=32,
            image_size=32,
            patch_size=8,
            max_text_len=8,
            d_model=16,
            embed_dim=8,
            vision_layers=1,
            vision_heads=4,
            text_layers=1,
            text_heads=4,
            fusion_layers=1,
            fusion_heads=4,
            d_ff=32,
        )
        image = torch.randn(3, 3, 32, 32)
        input_ids = torch.randint(3, 32, (3, 6))
        input_ids[:, 0] = 1
        labels = input_ids.clone()
        batch = model.make_multimodal_batch(image, input_ids, labels=labels)
        image_tokens = model.encode_image_tokens(image)
        text_tokens = model.encode_text_tokens(input_ids)
        fused = model.fuse(image, input_ids, causal=True)
        out = model.predict(batch)
        loss = model.loss_step(batch)
        loss.backward()
        self.assertEqual(image_tokens.shape, (3, 17, 16))
        self.assertEqual(text_tokens.shape, (3, 6, 16))
        self.assertEqual(fused.shape, (3, 6, 16))
        self.assertEqual(out["image_embeds"].shape, (3, 8))
        self.assertEqual(out["text_embeds"].shape, (3, 8))
        self.assertEqual(out["logits_per_image"].shape, (3, 3))
        self.assertEqual(out["itm_logits"].shape, (3, 2))
        self.assertEqual(out["lm_logits"].shape, (3, 6, 32))
        self.assertEqual(loss.dim(), 0)

    def test_blip_from_registry(self):
        model = create_model(
            "BLIP",
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
            fusion_layers=1,
            fusion_heads=4,
            d_ff=32,
        )
        image = torch.randn(2, 3, 32, 32)
        input_ids = torch.randint(1, 16, (2, 5))
        out = model(image, input_ids)
        self.assertEqual(out["multimodal_embeds"].shape, (2, 5, 16))
        self.assertEqual(out["lm_logits"].shape, (2, 5, 16))

    def test_blip2_qformer_bridge_api(self):
        model = tl.BLIP2(
            vocab_size=32,
            image_size=32,
            patch_size=8,
            max_text_len=8,
            d_model=16,
            embed_dim=8,
            vision_layers=1,
            vision_heads=4,
            qformer_layers=1,
            qformer_heads=4,
            num_query_tokens=4,
            lm_dim=12,
            d_ff=32,
        )
        image = torch.randn(3, 3, 32, 32)
        input_ids = torch.randint(3, 32, (3, 6))
        input_ids[:, 0] = 1
        labels = input_ids.clone()
        batch = model.make_multimodal_batch(image, input_ids, labels=labels)
        image_tokens = model.encode_image_tokens(image)
        qformer = model.qformer_features(image)
        prefix = model.language_prefix(image)
        logits = model.caption_logits(image, input_ids)
        out = model.predict(batch)
        loss = model.loss_step(batch)
        loss.backward()
        self.assertEqual(image_tokens.shape, (3, 17, 16))
        self.assertEqual(qformer.shape, (3, 4, 16))
        self.assertEqual(prefix.shape, (3, 4, 12))
        self.assertEqual(logits.shape, (3, 6, 32))
        self.assertEqual(out["image_embeds"].shape, (3, 8))
        self.assertEqual(out["text_embeds"].shape, (3, 8))
        self.assertEqual(out["logits_per_image"].shape, (3, 3))
        self.assertEqual(out["itm_logits"].shape, (3, 2))
        self.assertEqual(out["lm_logits"].shape, (3, 6, 32))
        self.assertEqual(loss.dim(), 0)

    def test_blip2_from_registry(self):
        model = create_model(
            "BLIP2",
            domain="multimodal",
            vocab_size=16,
            image_size=32,
            patch_size=8,
            max_text_len=8,
            d_model=16,
            embed_dim=8,
            vision_layers=1,
            vision_heads=4,
            qformer_layers=1,
            qformer_heads=4,
            num_query_tokens=4,
            lm_dim=12,
            d_ff=32,
        )
        image = torch.randn(2, 3, 32, 32)
        input_ids = torch.randint(1, 16, (2, 5))
        out = model(image, input_ids)
        self.assertEqual(out["qformer_embeds"].shape, (2, 4, 16))
        self.assertEqual(out["language_prefix"].shape, (2, 4, 12))
        self.assertEqual(out["lm_logits"].shape, (2, 5, 16))

    def test_llava_visual_instruction_language_model_api(self):
        model = tl.LLaVA(
            vocab_size=32,
            image_size=32,
            patch_size=8,
            max_text_len=8,
            d_model=16,
            embed_dim=16,
            vision_layers=1,
            vision_heads=4,
            decoder_layers=1,
            decoder_heads=4,
            d_ff=32,
            projector_hidden_dim=16,
        )
        image = torch.randn(2, 3, 32, 32)
        input_ids = torch.randint(3, 32, (2, 5))
        input_ids[:, 0] = 1
        labels = input_ids.clone()
        batch = model.make_multimodal_batch(image, input_ids, labels=labels)
        image_tokens = model.encode_image_tokens(image)
        projected = model.project_image_tokens(image_tokens)
        hidden = model.forward_features(image, input_ids)
        out = model.predict(batch)
        loss = model.loss_step(batch)
        generated = model.generate(image, input_ids[:, :2], max_new_tokens=2)
        loss.backward()
        self.assertEqual(image_tokens.shape, (2, 17, 16))
        self.assertEqual(projected.shape, (2, 16, 16))
        self.assertEqual(hidden.shape, (2, 21, 16))
        self.assertEqual(out["text_embeds"].shape, (2, 5, 16))
        self.assertEqual(out["lm_logits"].shape, (2, 5, 32))
        self.assertEqual(loss.dim(), 0)
        self.assertEqual(generated.shape[0], 2)
        self.assertLessEqual(generated.shape[1], 4)

    def test_llava_from_registry(self):
        model = create_model(
            "LLaVA",
            domain="multimodal",
            vocab_size=16,
            image_size=32,
            patch_size=8,
            max_text_len=8,
            d_model=16,
            embed_dim=16,
            vision_layers=1,
            vision_heads=4,
            decoder_layers=1,
            decoder_heads=4,
            d_ff=32,
            projector_hidden_dim=16,
        )
        image = torch.randn(2, 3, 32, 32)
        input_ids = torch.randint(1, 16, (2, 5))
        out = model(image, input_ids)
        self.assertEqual(out["multimodal_embeds"].shape, (2, 21, 16))
        self.assertEqual(out["lm_logits"].shape, (2, 5, 16))


if __name__ == "__main__":
    unittest.main()
