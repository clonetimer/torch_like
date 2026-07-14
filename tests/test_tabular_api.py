import unittest

import torch

import torch_like as tl
from torch_like.models import create_model


class TabularApiTests(unittest.TestCase):
    def test_tabnet_predict_loss_and_explain_api(self):
        model = tl.TabNet(
            input_dim=4,
            output_dim=3,
            d_model=8,
            categories=(5, 7),
            cat_embedding_dim=4,
            n_steps=2,
            gamma=1.5,
        )
        x_cont = torch.randn(6, 4)
        x_cat = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [0, 6]])
        y = torch.tensor([0, 1, 2, 1, 0, 2])
        batch = model.make_tabular_batch(x_cont, x_cat=x_cat, y=y)
        logits = model.predict(batch)
        loss = model.loss_step(batch)
        explanation = model.explain(batch)
        loss.backward()
        self.assertEqual(logits.shape, (6, 3))
        self.assertEqual(len(explanation["masks"]), 2)
        self.assertEqual(explanation["masks"][0].shape, (6, 12))
        self.assertEqual(explanation["sparsity_loss"].dim(), 0)
        self.assertEqual(loss.dim(), 0)

    def test_tabnet_from_registry_regression_api(self):
        model = create_model(
            "TabNet",
            domain="tabular",
            input_dim=3,
            output_dim=1,
            d_model=8,
            n_steps=2,
        )
        x_cont = torch.randn(4, 3)
        y = torch.randn(4, 1)
        batch = model.make_tabular_batch(x_cont, y=y)
        self.assertEqual(model.predict(batch).shape, (4, 1))
        self.assertEqual(model.loss_step(batch).dim(), 0)

    def test_node_neural_oblivious_tree_api(self):
        model = tl.NODE(
            input_dim=4,
            output_dim=3,
            d_model=8,
            categories=(5,),
            cat_embedding_dim=4,
            num_trees=4,
            tree_depth=3,
        )
        x_cont = torch.randn(6, 4)
        x_cat = torch.tensor([[0], [1], [2], [3], [4], [0]])
        y = torch.tensor([0, 1, 2, 1, 0, 2])
        batch = model.make_tabular_batch(x_cont, x_cat=x_cat, y=y)
        features = model.forward_features(x_cont, x_cat)
        tree_outputs = model.tree_outputs(x_cont, x_cat)
        importance = model.feature_importance()
        logits = model.predict(batch)
        loss = model.loss_step(batch)
        loss.backward()
        self.assertEqual(features.shape, (6, 8))
        self.assertEqual(tree_outputs.shape, (6, 4, 3))
        self.assertEqual(importance.shape, (8,))
        self.assertTrue(torch.allclose(importance.sum(), torch.tensor(1.0), atol=1e-5))
        self.assertEqual(logits.shape, (6, 3))
        self.assertEqual(loss.dim(), 0)
        self.assertIsNotNone(model.ensemble.leaf_responses.grad)

    def test_node_from_registry_regression_api(self):
        model = create_model(
            "NODE",
            domain="tabular",
            input_dim=3,
            output_dim=1,
            d_model=8,
            num_trees=3,
            tree_depth=2,
        )
        batch = model.make_tabular_batch(torch.randn(4, 3), y=torch.randn(4, 1))
        self.assertEqual(model.predict(batch).shape, (4, 1))
        self.assertEqual(model.loss_step(batch).dim(), 0)

    def test_ft_transformer_predict_loss_and_features_api(self):
        model = tl.FTTransformer(
            input_dim=4,
            output_dim=2,
            d_model=8,
            categories=(5, 7),
            n_heads=2,
            n_layers=1,
            d_ff=16,
        )
        x_cont = torch.randn(5, 4)
        x_cat = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]])
        y = torch.tensor([0, 1, 0, 1, 1])
        batch = model.make_tabular_batch(x_cont, x_cat=x_cat, y=y)
        tokens = model.tokenize(x_cont, x_cat)
        features = model.forward_features(x_cont, x_cat)
        logits = model.predict(batch)
        loss = model.loss_step(batch)
        self.assertEqual(tokens.shape, (5, 7, 8))
        self.assertEqual(features.shape, (5, 8))
        self.assertEqual(logits.shape, (5, 2))
        self.assertEqual(loss.dim(), 0)

    def test_ft_transformer_from_registry(self):
        model = create_model(
            "FTTransformer",
            domain="tabular",
            input_dim=3,
            output_dim=1,
            d_model=8,
            n_heads=2,
            n_layers=1,
            d_ff=16,
        )
        batch = model.make_tabular_batch(torch.randn(4, 3), y=torch.randn(4, 1))
        self.assertEqual(model.predict(batch).shape, (4, 1))

    def test_tab_transformer_contextual_embedding_api(self):
        model = tl.TabTransformer(
            input_dim=3,
            output_dim=2,
            d_model=8,
            categories=(5, 7),
            n_heads=2,
            n_layers=1,
            d_ff=16,
        )
        x_cont = torch.randn(5, 3)
        x_cat = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]])
        y = torch.tensor([0, 1, 0, 1, 1])
        batch = model.make_tabular_batch(x_cont, x_cat=x_cat, y=y)
        contextual = model.contextual_embeddings(x_cat)
        features = model.forward_features(x_cont, x_cat)
        logits = model.predict(batch)
        loss = model.loss_step(batch)
        loss.backward()
        self.assertEqual(contextual.shape, (5, 2, 8))
        self.assertEqual(features.shape, (5, 19))
        self.assertEqual(logits.shape, (5, 2))
        self.assertEqual(loss.dim(), 0)

    def test_tab_transformer_from_registry(self):
        model = create_model(
            "TabTransformer",
            domain="tabular",
            input_dim=2,
            output_dim=1,
            d_model=8,
            categories=(5,),
            n_heads=2,
            n_layers=1,
            d_ff=16,
        )
        batch = model.make_tabular_batch(
            torch.randn(4, 2), x_cat=torch.tensor([[0], [1], [2], [3]]), y=torch.randn(4, 1)
        )
        self.assertEqual(model.predict(batch).shape, (4, 1))

    def test_saint_row_column_attention_api(self):
        model = tl.SAINT(
            input_dim=3,
            output_dim=2,
            d_model=8,
            categories=(5, 7),
            n_heads=2,
            n_layers=1,
            d_ff=16,
        )
        x_cont = torch.randn(5, 3)
        x_cat = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]])
        y = torch.tensor([0, 1, 0, 1, 1])
        batch = model.make_tabular_batch(x_cont, x_cat=x_cat, y=y)
        tokens = model.tokenize(x_cont, x_cat)
        features = model.forward_features(x_cont, x_cat)
        logits = model.predict(batch)
        loss = model.loss_step(batch)
        loss.backward()
        self.assertEqual(tokens.shape, (5, 6, 8))
        self.assertEqual(features.shape, (5, 8))
        self.assertEqual(logits.shape, (5, 2))
        self.assertEqual(loss.dim(), 0)

    def test_saint_from_registry(self):
        model = create_model(
            "SAINT",
            domain="tabular",
            input_dim=2,
            output_dim=1,
            d_model=8,
            n_heads=2,
            n_layers=1,
            d_ff=16,
            use_row_attention=False,
        )
        batch = model.make_tabular_batch(torch.randn(4, 2), y=torch.randn(4, 1))
        self.assertEqual(model.predict(batch).shape, (4, 1))


if __name__ == "__main__":
    unittest.main()
