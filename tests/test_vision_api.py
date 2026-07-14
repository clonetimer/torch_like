import unittest

import torch

import torch_like as tl
from torch_like.models import create_model


class VisionApiTests(unittest.TestCase):
    def test_simclr_self_supervised_api(self):
        model = tl.SimCLR(
            in_channels=1,
            image_size=16,
            d_model=16,
            projection_dim=8,
            encoder_channels=(4, 8),
            temperature=0.2,
        )
        view1 = torch.randn(4, 1, 16, 16)
        view2 = torch.randn(4, 1, 16, 16)
        features = model.encode_image(view1)
        projected = model(view1)
        loss = model.loss_step(model.make_vision_batch(view1, y=view2))
        self.assertEqual(features.shape, (4, 16))
        self.assertEqual(projected.shape, (4, 8))
        self.assertEqual(loss.dim(), 0)

    def test_simclr_from_registry(self):
        model = create_model(
            "SimCLR",
            domain="vision",
            in_channels=1,
            image_size=16,
            d_model=16,
            projection_dim=8,
            encoder_channels=(4, 8),
        )
        out = model(torch.randn(2, 1, 16, 16))
        self.assertEqual(out.shape, (2, 8))

    def test_byol_self_supervised_api_and_target_update(self):
        model = tl.BYOL(
            in_channels=1,
            image_size=16,
            d_model=16,
            projection_dim=8,
            hidden_dim=16,
            encoder_channels=(4, 8),
            momentum=0.5,
        )
        view1 = torch.randn(4, 1, 16, 16)
        view2 = torch.randn(4, 1, 16, 16)
        features = model.encode_image(view1)
        projected = model(view1)
        loss = model.loss_step(model.make_vision_batch(view1, y=view2))
        before = next(model.target.parameters()).detach().clone()
        with torch.no_grad():
            next(model.online.parameters()).add_(1.0)
        model.update_target()
        after = next(model.target.parameters()).detach()
        self.assertEqual(features.shape, (4, 16))
        self.assertEqual(projected.shape, (4, 8))
        self.assertEqual(loss.dim(), 0)
        self.assertGreater((after - before).abs().sum().item(), 0.0)

    def test_byol_from_registry(self):
        model = create_model(
            "BYOL",
            domain="vision",
            in_channels=1,
            image_size=16,
            d_model=16,
            projection_dim=8,
            hidden_dim=16,
            encoder_channels=(4, 8),
        )
        out = model(torch.randn(2, 1, 16, 16))
        self.assertEqual(out.shape, (2, 8))

    def test_dino_self_distillation_api(self):
        model = tl.DINO(
            in_channels=1,
            image_size=16,
            patch_size=4,
            d_model=16,
            n_heads=4,
            e_layers=1,
            d_ff=32,
            projection_dim=8,
            hidden_dim=16,
            student_temp=0.2,
            teacher_temp=0.1,
            momentum=0.5,
        )
        student_view = torch.randn(4, 1, 16, 16)
        teacher_view = torch.randn(4, 1, 16, 16)
        logits = model(student_view)
        teacher_logits = model.teacher_logits(teacher_view)
        loss = model.loss_step(model.make_vision_batch(student_view, y=teacher_view))
        before = next(model.teacher.parameters()).detach().clone()
        with torch.no_grad():
            next(model.student.parameters()).add_(1.0)
        model.update_teacher()
        model.update_center(teacher_logits)
        after = next(model.teacher.parameters()).detach()
        self.assertEqual(logits.shape, (4, 8))
        self.assertEqual(teacher_logits.shape, (4, 8))
        self.assertEqual(loss.dim(), 0)
        self.assertEqual(model.center.shape, (1, 8))
        self.assertGreater((after - before).abs().sum().item(), 0.0)

    def test_dino_from_registry(self):
        model = create_model(
            "DINO",
            domain="vision",
            in_channels=1,
            image_size=16,
            patch_size=4,
            d_model=16,
            n_heads=4,
            e_layers=1,
            d_ff=32,
            projection_dim=8,
            hidden_dim=16,
        )
        out = model(torch.randn(2, 1, 16, 16))
        self.assertEqual(out.shape, (2, 8))

    def test_moco_momentum_contrast_api(self):
        model = tl.MoCo(
            in_channels=1,
            image_size=16,
            d_model=16,
            projection_dim=8,
            queue_size=8,
            encoder_channels=(4, 8),
            momentum=0.5,
            temperature=0.2,
        )
        query = torch.randn(4, 1, 16, 16)
        key = torch.randn(4, 1, 16, 16)
        features = model.encode_image(query)
        projected = model(query)
        logits, labels = model.contrastive_logits(query, key)
        loss = model.loss_step(model.make_vision_batch(query, y=key))
        self.assertEqual(features.shape, (4, 16))
        self.assertEqual(projected.shape, (4, 8))
        self.assertEqual(logits.shape, (4, 9))
        self.assertEqual(labels.shape, (4,))
        self.assertEqual(model.queue.shape, (8, 8))
        self.assertEqual(loss.dim(), 0)

    def test_moco_from_registry(self):
        model = create_model(
            "MoCo",
            domain="vision",
            in_channels=1,
            image_size=16,
            d_model=16,
            projection_dim=8,
            queue_size=8,
            encoder_channels=(4, 8),
        )
        out = model(torch.randn(2, 1, 16, 16))
        self.assertEqual(out.shape, (2, 8))

    def test_sam_promptable_segmentation_api(self):
        model = tl.SAM(
            in_channels=3,
            image_size=32,
            patch_size=8,
            d_model=16,
            encoder_layers=1,
            encoder_heads=4,
            d_ff=32,
            decoder_layers=1,
            decoder_heads=4,
            num_multimask_outputs=2,
            iou_head_hidden_dim=16,
        )
        x = torch.randn(2, 3, 32, 32)
        points = torch.tensor([[[0.5, 0.5], [0.25, 0.25]], [[0.4, 0.6], [0.8, 0.2]]])
        point_labels = torch.tensor([[1, 0], [1, -1]])
        boxes = torch.tensor([[0.1, 0.1, 0.8, 0.8], [0.2, 0.2, 0.7, 0.7]])
        out = model(x, points=points, point_labels=point_labels, boxes=boxes, multimask_output=True)
        image_embeddings = model.encode_image(x)
        sparse, dense = model.encode_prompts(
            batch_size=2,
            device=x.device,
            dtype=x.dtype,
            points=points,
            point_labels=point_labels,
            boxes=boxes,
        )
        decoded = model.predict_masks(image_embeddings, sparse, dense, multimask_output=False)
        self.assertEqual(image_embeddings.shape, (2, 16, 4, 4))
        self.assertEqual(sparse.shape, (2, 4, 16))
        self.assertEqual(dense.shape, (2, 16, 4, 4))
        self.assertEqual(out["masks"].shape, (2, 2, 32, 32))
        self.assertEqual(out["low_res_masks"].shape, (2, 2, 16, 16))
        self.assertEqual(out["iou_predictions"].shape, (2, 2))
        self.assertEqual(decoded["low_res_masks"].shape, (2, 1, 16, 16))
        self.assertEqual(decoded["iou_predictions"].shape, (2, 1))

    def test_sam_from_registry_with_mask_prompt(self):
        model = create_model(
            "SAM",
            domain="vision",
            in_channels=1,
            image_size=32,
            patch_size=8,
            d_model=16,
            encoder_layers=1,
            encoder_heads=4,
            d_ff=32,
            decoder_layers=1,
            decoder_heads=4,
            num_multimask_outputs=2,
            iou_head_hidden_dim=16,
            prompt_coordinate_format="pixel",
        )
        x = torch.randn(2, 1, 32, 32)
        mask_prompt = torch.randn(2, 1, 32, 32)
        points = torch.tensor([[[16.0, 16.0]], [[8.0, 24.0]]])
        out = model.predict(model.make_vision_batch(x))
        prompted = model(x, points=points, masks=mask_prompt, multimask_output=False)
        self.assertEqual(out["masks"].shape, (2, 2, 32, 32))
        self.assertEqual(prompted["masks"].shape, (2, 1, 32, 32))
        self.assertEqual(prompted["iou_predictions"].shape, (2, 1))

    def test_mae_self_supervised_api(self):
        model = tl.MAE(
            image_size=32,
            patch_size=8,
            d_model=16,
            encoder_layers=1,
            encoder_heads=4,
            d_ff=32,
            decoder_dim=8,
            decoder_layers=1,
            decoder_heads=2,
            decoder_d_ff=16,
        )
        batch = model.make_vision_batch(torch.randn(2, 3, 32, 32))
        output = model.predict(batch)
        loss = model.fit_step(batch)
        loss.backward()
        self.assertEqual(output["reconstruction"].shape, (2, 16, 192))
        self.assertEqual(output["mask"].shape, (2, 16))
        self.assertEqual(model.encode_image(batch.x).shape, (2, 16, 16))
        self.assertTrue(torch.isfinite(loss))

    def test_resnet_predict_api(self):
        model = tl.ResNet.resnet18(in_channels=3, num_classes=5, d_model=8)
        x = torch.randn(2, 3, 64, 64)
        batch = model.make_vision_batch(x)
        out = model.predict(batch)
        self.assertEqual(out.shape, (2, 5))

    def test_resnet_from_registry(self):
        model = create_model(
            "ResNet",
            domain="vision",
            in_channels=1,
            num_classes=3,
            d_model=8,
            block="basic",
            layers=(1, 1, 1, 1),
        )
        x = torch.randn(2, 1, 32, 32)
        features = model.forward_features(x)
        out = model.predict(model.make_vision_batch(x))
        self.assertEqual(features.shape, (2, 64))
        self.assertEqual(out.shape, (2, 3))

    def test_resnet_bottleneck_api(self):
        model = tl.ResNet.resnet50(in_channels=3, num_classes=7, d_model=8)
        x = torch.randn(2, 3, 64, 64)
        out = model.predict(model.make_vision_batch(x))
        self.assertEqual(out.shape, (2, 7))

    def test_vit_predict_api(self):
        model = tl.ViT(
            in_channels=3,
            num_classes=5,
            image_size=32,
            patch_size=8,
            d_model=24,
            n_heads=3,
            e_layers=2,
            d_ff=48,
        )
        x = torch.randn(2, 3, 32, 32)
        out = model.predict(model.make_vision_batch(x))
        self.assertEqual(out.shape, (2, 5))

    def test_vit_from_registry_mean_pool(self):
        model = create_model(
            "ViT",
            domain="vision",
            in_channels=1,
            num_classes=3,
            image_size=32,
            patch_size=8,
            d_model=16,
            n_heads=4,
            e_layers=1,
            d_ff=32,
            pool="mean",
        )
        x = torch.randn(2, 1, 32, 32)
        features = model.forward_features(x)
        out = model.predict(model.make_vision_batch(x))
        self.assertEqual(features.shape, (2, 16))
        self.assertEqual(out.shape, (2, 3))

    def test_deit_predict_api(self):
        model = tl.DeiT(
            in_channels=3,
            num_classes=5,
            image_size=32,
            patch_size=8,
            d_model=24,
            n_heads=3,
            e_layers=2,
            d_ff=48,
            distillation=True,
        )
        x = torch.randn(2, 3, 32, 32)
        cls_logits, dist_logits = model.forward_logits(x)
        out = model.predict(model.make_vision_batch(x))
        self.assertEqual(cls_logits.shape, (2, 5))
        self.assertEqual(dist_logits.shape, (2, 5))
        self.assertEqual(out.shape, (2, 5))

    def test_deit_from_registry_without_distillation(self):
        model = create_model(
            "DeiT",
            domain="vision",
            in_channels=1,
            num_classes=3,
            image_size=32,
            patch_size=8,
            d_model=16,
            n_heads=4,
            e_layers=1,
            d_ff=32,
            distillation=False,
            output_mode="cls",
        )
        x = torch.randn(2, 1, 32, 32)
        features = model.forward_features(x)
        out = model.predict(model.make_vision_batch(x))
        self.assertEqual(features.shape, (2, 16))
        self.assertEqual(out.shape, (2, 3))

    def test_unet_predict_api(self):
        model = tl.UNet(in_channels=3, num_classes=2, base_channels=8, depth=3)
        x = torch.randn(2, 3, 32, 32)
        out = model.predict(model.make_vision_batch(x))
        self.assertEqual(out.shape, (2, 2, 32, 32))

    def test_unet_from_registry(self):
        model = create_model(
            "UNet",
            domain="vision",
            in_channels=1,
            num_classes=4,
            base_channels=4,
            depth=3,
        )
        x = torch.randn(2, 1, 32, 32)
        features = model.forward_features(x)
        out = model.predict(model.make_vision_batch(x))
        self.assertEqual(features.shape, (2, 4, 32, 32))
        self.assertEqual(out.shape, (2, 4, 32, 32))

    def test_deeplabv3_predict_api(self):
        model = tl.DeepLabV3(
            in_channels=3,
            num_classes=2,
            backbone_channels=(8, 16, 32),
            aspp_channels=8,
            atrous_rates=(2, 4),
        )
        x = torch.randn(2, 3, 32, 32)
        out = model.predict(model.make_vision_batch(x))
        self.assertEqual(out.shape, (2, 2, 32, 32))

    def test_deeplabv3_from_registry(self):
        model = create_model(
            "DeepLabV3",
            domain="vision",
            in_channels=1,
            num_classes=3,
            backbone_channels=(4, 8, 16),
            aspp_channels=6,
            atrous_rates=(1, 2),
        )
        x = torch.randn(2, 1, 32, 32)
        features = model.forward_features(x)
        out = model.predict(model.make_vision_batch(x))
        self.assertEqual(features.shape, (2, 6, 8, 8))
        self.assertEqual(out.shape, (2, 3, 32, 32))

    def test_detr_predict_api(self):
        model = tl.DETR(
            in_channels=3,
            num_classes=5,
            d_model=16,
            num_queries=4,
            backbone_channels=(8, 16),
            n_heads=4,
            e_layers=1,
            d_layers=1,
            d_ff=32,
        )
        x = torch.randn(2, 3, 32, 32)
        out = model.predict(model.make_vision_batch(x))
        detections = model.post_process(out, top_k=2)
        self.assertEqual(out["pred_logits"].shape, (2, 4, 6))
        self.assertEqual(out["pred_boxes"].shape, (2, 4, 4))
        self.assertEqual(len(detections), 2)
        self.assertLessEqual(detections[0]["scores"].numel(), 2)

    def test_detr_from_registry(self):
        model = create_model(
            "DETR",
            domain="vision",
            in_channels=1,
            num_classes=3,
            d_model=16,
            num_queries=5,
            backbone_channels=(8, 16),
            n_heads=4,
            e_layers=1,
            d_layers=1,
            d_ff=32,
        )
        x = torch.randn(2, 1, 32, 32)
        features = model.forward_features(x)
        out = model.predict(model.make_vision_batch(x))
        self.assertEqual(features.shape, (2, 5, 16))
        self.assertEqual(out["pred_logits"].shape, (2, 5, 4))
        self.assertEqual(out["pred_boxes"].shape, (2, 5, 4))

    def test_rtdetr_query_selection_and_registry_api(self):
        model = create_model(
            "RTDETR",
            domain="vision",
            in_channels=1,
            num_classes=3,
            image_size=32,
            d_model=16,
            num_queries=5,
            backbone_channels=(8, 16, 32),
            n_heads=4,
            encoder_layers=1,
            decoder_layers=1,
            d_ff=32,
        )
        x = torch.randn(2, 1, 32, 32)
        memory = model.encode_image(x)
        features = model.forward_features(x)
        out = model.predict(model.make_vision_batch(x))
        detections = model.post_process(out, top_k=2)
        self.assertEqual(memory.shape, (2, 16 * 16 + 8 * 8 + 4 * 4, 16))
        self.assertEqual(features["query_features"].shape, (2, 5, 16))
        self.assertEqual(out["pred_logits"].shape, (2, 5, 3))
        self.assertEqual(out["pred_boxes"].shape, (2, 5, 4))
        self.assertEqual(out["selected_indices"].shape, (2, 5))
        self.assertEqual(len(detections), 2)
        self.assertLessEqual(detections[0]["scores"].numel(), 2)

    def test_yolo_predict_and_post_process_api(self):
        model = tl.YOLO(
            in_channels=3,
            num_classes=5,
            d_model=16,
            anchors=((0.10, 0.10), (0.20, 0.20)),
            backbone_channels=(8, 16),
            grid_size=4,
        )
        x = torch.randn(2, 3, 32, 32)
        raw = model.raw_predictions(x)
        out = model.predict(model.make_vision_batch(x))
        detections = model.post_process(out, top_k=3)
        self.assertEqual(raw.shape, (2, 2, 4, 4, 10))
        self.assertEqual(out["pred_logits"].shape, (2, 32, 5))
        self.assertEqual(out["pred_boxes"].shape, (2, 32, 4))
        self.assertEqual(out["objectness"].shape, (2, 32))
        self.assertEqual(len(detections), 2)
        self.assertLessEqual(detections[0]["scores"].numel(), 3)

    def test_yolo_from_registry(self):
        model = create_model(
            "YOLO",
            domain="vision",
            in_channels=1,
            num_classes=3,
            d_model=16,
            anchors=((0.10, 0.10), (0.20, 0.20)),
            backbone_channels=(8, 16),
            grid_size=4,
        )
        x = torch.randn(2, 1, 32, 32)
        features = model.forward_features(x)
        out = model.predict(model.make_vision_batch(x))
        self.assertEqual(features.shape, (2, 16, 4, 4))
        self.assertEqual(out["pred_logits"].shape, (2, 32, 3))
        self.assertEqual(out["pred_boxes"].shape, (2, 32, 4))

    def test_grounding_dino_open_vocabulary_detection_api(self):
        model = tl.GroundingDINO(
            in_channels=3,
            image_size=32,
            d_model=16,
            vocab_size=32,
            max_text_len=8,
            num_queries=4,
            backbone_channels=(8, 16),
            n_heads=4,
            e_layers=1,
            d_layers=1,
            text_layers=1,
            d_ff=32,
        )
        image = torch.randn(2, 3, 32, 32)
        input_ids = torch.randint(1, 32, (2, 6))
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        features = model.forward_features(image, input_ids, attention_mask)
        out = model(image, input_ids, attention_mask)
        detections = model.post_process(out, top_k=2)
        self.assertEqual(features["image_tokens"].shape, (2, 256, 16))
        self.assertEqual(features["text_tokens"].shape, (2, 6, 16))
        self.assertEqual(features["query_features"].shape, (2, 4, 16))
        self.assertEqual(out["token_logits"].shape, (2, 4, 6))
        self.assertEqual(out["objectness"].shape, (2, 4))
        self.assertEqual(out["pred_boxes"].shape, (2, 4, 4))
        self.assertEqual(len(detections), 2)
        self.assertLessEqual(detections[0]["scores"].numel(), 2)

    def test_grounding_dino_from_registry(self):
        model = create_model(
            "GroundingDINO",
            domain="vision",
            in_channels=1,
            image_size=32,
            d_model=16,
            vocab_size=24,
            max_text_len=8,
            num_queries=5,
            backbone_channels=(8, 16),
            n_heads=4,
            e_layers=1,
            d_layers=1,
            text_layers=1,
            d_ff=32,
        )
        image = torch.randn(2, 1, 32, 32)
        input_ids = torch.randint(1, 24, (2, 5))
        out = model(image, input_ids)
        self.assertEqual(out["token_logits"].shape, (2, 5, 5))
        self.assertEqual(out["pred_boxes"].shape, (2, 5, 4))

    def test_mask2former_universal_segmentation_api(self):
        model = tl.Mask2Former(
            in_channels=3,
            num_classes=4,
            image_size=32,
            d_model=16,
            num_queries=5,
            backbone_channels=(8, 16),
            n_heads=4,
            decoder_layers=1,
            d_ff=32,
            mask_dim=8,
        )
        x = torch.randn(2, 3, 32, 32)
        features = model.forward_features(x)
        masks = model.predict_masks(features["query_features"], features["mask_features"], output_size=(32, 32))
        out = model.predict(model.make_vision_batch(x))
        detections = model.post_process(out, top_k=2)
        self.assertEqual(features["query_features"].shape, (2, 5, 16))
        self.assertEqual(features["mask_features"].shape, (2, 8, 16, 16))
        self.assertEqual(masks.shape, (2, 5, 32, 32))
        self.assertEqual(out["pred_logits"].shape, (2, 5, 5))
        self.assertEqual(out["pred_masks"].shape, (2, 5, 32, 32))
        self.assertEqual(out["semantic_logits"].shape, (2, 4, 32, 32))
        self.assertEqual(len(detections), 2)
        self.assertLessEqual(detections[0]["scores"].numel(), 2)

    def test_mask2former_from_registry(self):
        model = create_model(
            "Mask2Former",
            domain="vision",
            in_channels=1,
            num_classes=3,
            image_size=32,
            d_model=16,
            num_queries=4,
            backbone_channels=(8, 16),
            n_heads=4,
            decoder_layers=1,
            d_ff=32,
            mask_dim=8,
        )
        x = torch.randn(2, 1, 32, 32)
        out = model.predict(model.make_vision_batch(x))
        self.assertEqual(out["pred_logits"].shape, (2, 4, 4))
        self.assertEqual(out["pred_masks"].shape, (2, 4, 32, 32))
        self.assertEqual(out["semantic_logits"].shape, (2, 3, 32, 32))

    def test_fcn_predict_api(self):
        model = tl.FCN(in_channels=3, num_classes=2, channels=(8, 16), strides=(2, 2))
        x = torch.randn(2, 3, 32, 32)
        out = model.predict(model.make_vision_batch(x))
        self.assertEqual(out.shape, (2, 2, 32, 32))

    def test_fcn_from_registry(self):
        model = create_model(
            "FCN",
            domain="vision",
            in_channels=1,
            num_classes=3,
            channels=(4, 8),
            strides=(2, 2),
        )
        x = torch.randn(2, 1, 32, 32)
        features = model.forward_features(x)
        out = model.predict(model.make_vision_batch(x))
        self.assertEqual(features.shape, (2, 8, 8, 8))
        self.assertEqual(out.shape, (2, 3, 32, 32))

    def test_densenet_predict_api(self):
        model = tl.DenseNet(
            in_channels=3,
            num_classes=5,
            growth_rate=4,
            block_layers=(2, 2),
            num_init_features=8,
        )
        x = torch.randn(2, 3, 32, 32)
        out = model.predict(model.make_vision_batch(x))
        self.assertEqual(out.shape, (2, 5))

    def test_densenet_from_registry(self):
        model = create_model(
            "DenseNet",
            domain="vision",
            in_channels=1,
            num_classes=3,
            growth_rate=4,
            block_layers=(2, 2, 2),
            num_init_features=8,
            compression=0.5,
        )
        x = torch.randn(2, 1, 32, 32)
        features = model.forward_features(x)
        out = model.predict(model.make_vision_batch(x))
        self.assertEqual(features.shape, (2, model.feature_dim))
        self.assertEqual(out.shape, (2, 3))

    def test_mobilenetv2_predict_api(self):
        model = tl.MobileNetV2(
            in_channels=3,
            num_classes=5,
            width_mult=0.25,
            d_model=16,
            last_channel=64,
            inverted_residual_setting=((1, 8, 1, 1), (6, 16, 1, 2)),
        )
        x = torch.randn(2, 3, 32, 32)
        out = model.predict(model.make_vision_batch(x))
        self.assertEqual(out.shape, (2, 5))

    def test_mobilenetv2_from_registry(self):
        model = create_model(
            "MobileNetV2",
            domain="vision",
            in_channels=1,
            num_classes=3,
            width_mult=0.25,
            d_model=16,
            last_channel=64,
            inverted_residual_setting=((1, 8, 1, 1), (6, 16, 1, 2)),
        )
        x = torch.randn(2, 1, 32, 32)
        features = model.forward_features(x)
        out = model.predict(model.make_vision_batch(x))
        self.assertEqual(features.shape, (2, model.feature_dim))
        self.assertEqual(out.shape, (2, 3))

    def test_efficientnet_predict_api(self):
        model = tl.EfficientNet(
            in_channels=3,
            num_classes=5,
            width_mult=0.25,
            depth_mult=0.25,
            d_model=16,
            last_channel=64,
            block_settings=((1, 8, 1, 1, 3), (6, 16, 1, 2, 3)),
        )
        x = torch.randn(2, 3, 32, 32)
        out = model.predict(model.make_vision_batch(x))
        self.assertEqual(out.shape, (2, 5))

    def test_efficientnet_from_registry(self):
        model = create_model(
            "EfficientNet",
            domain="vision",
            in_channels=1,
            num_classes=3,
            width_mult=0.25,
            depth_mult=0.25,
            d_model=16,
            last_channel=64,
            block_settings=((1, 8, 1, 1, 3), (6, 16, 1, 2, 3)),
            drop_connect_rate=0.0,
        )
        x = torch.randn(2, 1, 32, 32)
        features = model.forward_features(x)
        out = model.predict(model.make_vision_batch(x))
        self.assertEqual(features.shape, (2, model.feature_dim))
        self.assertEqual(out.shape, (2, 3))

    def test_swin_transformer_predict_api(self):
        model = tl.SwinTransformer(
            in_channels=3,
            num_classes=5,
            image_size=32,
            patch_size=4,
            embed_dim=16,
            depths=(1, 1),
            num_heads=(2, 4),
            window_size=4,
        )
        x = torch.randn(2, 3, 32, 32)
        out = model.predict(model.make_vision_batch(x))
        self.assertEqual(out.shape, (2, 5))

    def test_swin_transformer_from_registry(self):
        model = create_model(
            "SwinTransformer",
            domain="vision",
            in_channels=1,
            num_classes=3,
            image_size=32,
            patch_size=4,
            embed_dim=16,
            depths=(1, 1, 1),
            num_heads=(2, 4, 8),
            window_size=4,
        )
        x = torch.randn(2, 1, 32, 32)
        features = model.forward_features(x)
        out = model.predict(model.make_vision_batch(x))
        self.assertEqual(features.shape, (2, model.feature_dim))
        self.assertEqual(out.shape, (2, 3))

    def test_convnext_predict_api(self):
        model = tl.ConvNeXt(
            in_channels=3,
            num_classes=5,
            depths=(1, 1),
            dims=(8, 16),
            layer_scale_init_value=1e-6,
        )
        x = torch.randn(2, 3, 32, 32)
        out = model.predict(model.make_vision_batch(x))
        self.assertEqual(out.shape, (2, 5))

    def test_convnext_from_registry(self):
        model = create_model(
            "ConvNeXt",
            domain="vision",
            in_channels=1,
            num_classes=3,
            depths=(1, 1, 1),
            dims=(8, 16, 32),
        )
        x = torch.randn(2, 1, 32, 32)
        features = model.forward_features(x)
        out = model.predict(model.make_vision_batch(x))
        self.assertEqual(features.shape, (2, model.feature_dim))
        self.assertEqual(out.shape, (2, 3))

    def test_googlenet_predict_api(self):
        model = tl.GoogLeNet(
            in_channels=3,
            num_classes=5,
            stem_channels=8,
            inception_channels=((8, 8, 8, 4, 4, 4), (8, 8, 12, 4, 4, 4)),
        )
        x = torch.randn(2, 3, 32, 32)
        out = model.predict(model.make_vision_batch(x))
        self.assertEqual(out.shape, (2, 5))

    def test_googlenet_from_registry(self):
        model = create_model(
            "GoogLeNet",
            domain="vision",
            in_channels=1,
            num_classes=3,
            stem_channels=8,
            inception_channels=((8, 8, 8, 4, 4, 4), (8, 8, 12, 4, 4, 4)),
        )
        x = torch.randn(2, 1, 32, 32)
        features = model.forward_features(x)
        out = model.predict(model.make_vision_batch(x))
        self.assertEqual(features.shape, (2, model.feature_dim))
        self.assertEqual(out.shape, (2, 3))

    def test_inception_v3_predict_api(self):
        model = tl.InceptionV3(
            in_channels=3,
            num_classes=5,
            d_model=8,
            stem_channels=8,
            a_blocks=1,
            b_blocks=1,
            c_blocks=1,
        )
        x = torch.randn(2, 3, 64, 64)
        out = model.predict(model.make_vision_batch(x))
        self.assertEqual(out.shape, (2, 5))

    def test_inception_v3_from_registry(self):
        model = create_model(
            "InceptionV3",
            domain="vision",
            in_channels=1,
            num_classes=3,
            d_model=8,
            stem_channels=8,
            a_blocks=1,
            b_blocks=1,
            c_blocks=1,
        )
        x = torch.randn(2, 1, 64, 64)
        features = model.forward_features(x)
        out = model.predict(model.make_vision_batch(x))
        self.assertEqual(features.shape, (2, model.feature_dim))
        self.assertEqual(out.shape, (2, 3))

    def test_vgg_predict_api(self):
        model = tl.VGG(
            in_channels=3,
            num_classes=5,
            cfg=(8, "M", 16, "M"),
            classifier_dim=32,
            dropout=0.0,
        )
        x = torch.randn(2, 3, 32, 32)
        out = model.predict(model.make_vision_batch(x))
        self.assertEqual(out.shape, (2, 5))

    def test_vgg_from_registry(self):
        model = create_model(
            "VGG",
            domain="vision",
            in_channels=1,
            num_classes=3,
            cfg=(4, "M", 8, "M"),
            classifier_dim=16,
            batch_norm=True,
            dropout=0.0,
        )
        x = torch.randn(2, 1, 32, 32)
        features = model.forward_features(x)
        out = model.predict(model.make_vision_batch(x))
        self.assertEqual(features.shape, (2, model.feature_dim))
        self.assertEqual(out.shape, (2, 3))

    def test_alexnet_predict_api(self):
        model = tl.AlexNet(
            in_channels=3,
            num_classes=5,
            channels=(8, 16, 24, 24, 16),
            classifier_dim=32,
            dropout=0.0,
        )
        x = torch.randn(2, 3, 64, 64)
        out = model.predict(model.make_vision_batch(x))
        self.assertEqual(out.shape, (2, 5))

    def test_alexnet_from_registry(self):
        model = create_model(
            "AlexNet",
            domain="vision",
            in_channels=1,
            num_classes=3,
            channels=(4, 8, 12, 12, 8),
            classifier_dim=16,
            dropout=0.0,
        )
        x = torch.randn(2, 1, 64, 64)
        features = model.forward_features(x)
        out = model.predict(model.make_vision_batch(x))
        self.assertEqual(features.shape, (2, model.feature_dim))
        self.assertEqual(out.shape, (2, 3))

    def test_mlp_mixer_predict_api(self):
        model = tl.MLPMixer(
            in_channels=3,
            num_classes=5,
            image_size=32,
            patch_size=8,
            d_model=16,
            n_layers=2,
            token_dim=8,
            channel_dim=32,
        )
        x = torch.randn(2, 3, 32, 32)
        out = model.predict(model.make_vision_batch(x))
        self.assertEqual(out.shape, (2, 5))

    def test_mlp_mixer_from_registry(self):
        model = create_model(
            "MLPMixer",
            domain="vision",
            in_channels=1,
            num_classes=3,
            image_size=32,
            patch_size=8,
            d_model=12,
            n_layers=1,
            token_dim=6,
            channel_dim=24,
        )
        x = torch.randn(2, 1, 32, 32)
        features = model.forward_features(x)
        out = model.predict(model.make_vision_batch(x))
        self.assertEqual(features.shape, (2, model.feature_dim))
        self.assertEqual(out.shape, (2, 3))

    def test_resnext_predict_api(self):
        model = tl.ResNeXt(
            in_channels=3,
            num_classes=5,
            d_model=8,
            layers=(1, 1, 1, 1),
            groups=2,
            width_per_group=8,
        )
        x = torch.randn(2, 3, 64, 64)
        out = model.predict(model.make_vision_batch(x))
        self.assertEqual(out.shape, (2, 5))

    def test_resnext_from_registry(self):
        model = create_model(
            "ResNeXt",
            domain="vision",
            in_channels=1,
            num_classes=3,
            d_model=8,
            layers=(1, 1, 1, 1),
            groups=2,
            width_per_group=8,
        )
        x = torch.randn(2, 1, 64, 64)
        features = model.forward_features(x)
        out = model.predict(model.make_vision_batch(x))
        self.assertEqual(features.shape, (2, model.feature_dim))
        self.assertEqual(out.shape, (2, 3))

    def test_senet_predict_api(self):
        model = tl.SENet(
            in_channels=3,
            num_classes=5,
            d_model=8,
            layers=(1, 1, 1, 1),
            reduction=4,
        )
        x = torch.randn(2, 3, 64, 64)
        out = model.predict(model.make_vision_batch(x))
        self.assertEqual(out.shape, (2, 5))

    def test_senet_from_registry(self):
        model = create_model(
            "SENet",
            domain="vision",
            in_channels=1,
            num_classes=3,
            d_model=8,
            layers=(1, 1, 1, 1),
            reduction=4,
        )
        x = torch.randn(2, 1, 64, 64)
        features = model.forward_features(x)
        out = model.predict(model.make_vision_batch(x))
        self.assertEqual(features.shape, (2, model.feature_dim))
        self.assertEqual(out.shape, (2, 3))


if __name__ == "__main__":
    unittest.main()
