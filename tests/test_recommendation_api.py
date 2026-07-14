import unittest

import torch

import torch_like as tl
from torch_like.models import create_model


class RecommendationApiTests(unittest.TestCase):
    def test_ncf_predict_and_loss_api(self):
        model = tl.NCF(num_users=10, num_items=12, mf_dim=4, d_model=8, mlp_layers=(8, 4))
        user_ids = torch.tensor([0, 1, 2, 3])
        item_ids = torch.tensor([3, 2, 1, 0])
        y = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
        batch = model.make_recommendation_batch(user_ids=user_ids, item_ids=item_ids, y=y)
        logits = model.predict(batch)
        loss = model.loss_step(batch)
        loss.backward()
        self.assertEqual(logits.shape, (4, 1))
        self.assertEqual(loss.dim(), 0)

    def test_ncf_from_registry(self):
        model = create_model("NCF", domain="recommendation", num_users=5, num_items=6, mf_dim=4, d_model=8)
        batch = model.make_recommendation_batch(user_ids=torch.tensor([0, 1]), item_ids=torch.tensor([1, 2]))
        self.assertEqual(model.predict(batch).shape, (2, 1))

    def test_deepfm_ctr_api(self):
        model = tl.DeepFM(categories=(5, 7, 9), num_continuous=2, fm_dim=4, mlp_layers=(8, 4))
        x_cat = torch.tensor([[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]])
        x_cont = torch.randn(4, 2)
        y = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
        batch = model.make_recommendation_batch(x_cat=x_cat, x_cont=x_cont, y=y)
        logits = model.predict(batch)
        loss = model.loss_step(batch)
        self.assertEqual(logits.shape, (4, 1))
        self.assertEqual(loss.dim(), 0)

    def test_deepfm_from_registry_multiclass(self):
        model = create_model(
            "DeepFM",
            domain="recommendation",
            categories=(5, 7),
            output_dim=3,
            fm_dim=4,
            mlp_layers=(8,),
        )
        x_cat = torch.tensor([[0, 1], [1, 2], [2, 3]])
        y = torch.tensor([0, 1, 2])
        batch = model.make_recommendation_batch(x_cat=x_cat, y=y)
        self.assertEqual(model.predict(batch).shape, (3, 3))
        self.assertEqual(model.loss_step(batch).dim(), 0)

    def test_dcn_cross_network_api(self):
        model = tl.DCN(categories=(5, 7), num_continuous=3, embedding_dim=4, cross_layers=2, mlp_layers=(8,))
        x_cat = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 4]])
        x_cont = torch.randn(4, 3)
        y = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
        batch = model.make_recommendation_batch(x_cat=x_cat, x_cont=x_cont, y=y)
        logits = model.predict(batch)
        loss = model.loss_step(batch)
        self.assertEqual(logits.shape, (4, 1))
        self.assertEqual(loss.dim(), 0)

    def test_dcn_from_registry(self):
        model = create_model(
            "DCN",
            domain="recommendation",
            categories=(5,),
            num_continuous=2,
            embedding_dim=4,
            cross_layers=2,
            mlp_layers=(8,),
        )
        batch = model.make_recommendation_batch(x_cat=torch.tensor([[0], [1]]), x_cont=torch.randn(2, 2))
        self.assertEqual(model.predict(batch).shape, (2, 1))

    def test_lightgcn_graph_scoring_and_bpr_loss(self):
        model = tl.LightGCN(num_users=3, num_items=4, embedding_dim=6, num_layers=1)
        adjacency = torch.eye(7)
        user_ids = torch.tensor([0, 1, 2])
        item_ids = torch.tensor([1, 2, 3])
        batch = model.make_recommendation_batch(user_ids=user_ids, item_ids=item_ids)
        logits = model.predict(batch)
        full_scores = model.full_sort_scores(user_ids, adjacency=adjacency)
        loss = model.bpr_loss(
            user_ids=user_ids,
            positive_item_ids=torch.tensor([1, 2, 3]),
            negative_item_ids=torch.tensor([0, 1, 2]),
            adjacency=adjacency,
        )
        loss.backward()
        self.assertEqual(logits.shape, (3, 1))
        self.assertEqual(full_scores.shape, (3, 4))
        self.assertEqual(loss.dim(), 0)

    def test_lightgcn_from_registry(self):
        model = create_model("LightGCN", domain="recommendation", num_users=3, num_items=4, embedding_dim=6)
        scores = model(user_ids=torch.tensor([0, 1]), item_ids=torch.tensor([2, 3]))
        self.assertEqual(scores.shape, (2, 1))

    def test_sasrec_next_item_and_candidate_api(self):
        model = tl.SASRec(num_items=8, max_seq_len=5, d_model=8, n_heads=2, n_layers=1, d_ff=16)
        sequences = torch.tensor([[1, 2, 3, 0, 0], [2, 3, 4, 5, 0], [3, 4, 5, 6, 7]])
        labels = torch.tensor([4, 6, 8])
        batch = model.make_recommendation_batch(x_cat=sequences, y=labels)
        logits = model.predict(batch)
        candidate_scores = model(user_ids=None, item_ids=torch.tensor([4, 6, 8]), x_cat=sequences)
        loss = model.loss_step(batch)
        loss.backward()
        self.assertEqual(logits.shape, (3, 8))
        self.assertEqual(candidate_scores.shape, (3, 1))
        self.assertEqual(loss.dim(), 0)

    def test_sasrec_from_registry(self):
        model = create_model(
            "SASRec",
            domain="recommendation",
            num_items=8,
            max_seq_len=5,
            d_model=8,
            n_heads=2,
            n_layers=1,
            d_ff=16,
        )
        logits = model(x_cat=torch.tensor([[1, 2, 3, 0, 0], [2, 3, 4, 5, 0]]))
        self.assertEqual(logits.shape, (2, 8))

    def test_bert4rec_masked_item_prediction_api(self):
        model = tl.BERT4Rec(num_items=8, max_seq_len=5, d_model=8, n_heads=2, n_layers=1, d_ff=16)
        sequences = torch.tensor([[1, 2, 9, 0, 0], [2, 9, 4, 5, 0], [3, 4, 5, 9, 7]])
        labels = torch.tensor([[0, 0, 3, 0, 0], [0, 3, 0, 0, 0], [0, 0, 0, 6, 0]])
        mask_positions = sequences.eq(model.mask_token_id)
        batch = model.make_recommendation_batch(x_cat=sequences, y=labels)
        encoded = model.encode(sequences)
        logits = model.predict(batch)
        masked_logits = model.masked_item_logits(sequences, mask_positions)
        candidate_scores = model.score_items(sequences, torch.tensor([3, 3, 6]), torch.tensor([2, 1, 3]))
        loss = model.loss_step(batch)
        loss.backward()
        self.assertEqual(encoded.shape, (3, 5, 8))
        self.assertEqual(logits.shape, (3, 5, 8))
        self.assertEqual(masked_logits.shape, (3, 8))
        self.assertEqual(candidate_scores.shape, (3, 1))
        self.assertEqual(loss.dim(), 0)

    def test_bert4rec_from_registry(self):
        model = create_model(
            "BERT4Rec",
            domain="recommendation",
            num_items=8,
            max_seq_len=5,
            d_model=8,
            n_heads=2,
            n_layers=1,
            d_ff=16,
        )
        logits = model(x_cat=torch.tensor([[1, 2, 9, 0, 0], [2, 9, 4, 5, 0]]))
        self.assertEqual(logits.shape, (2, 5, 8))


if __name__ == "__main__":
    unittest.main()
