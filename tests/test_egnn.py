import unittest

import torch

import torch_like as tl
from torch_like.models.graph.egnn import EGNN, GeometricGraphBatch


class EGNNTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        self.model = EGNN(
            input_dim=4,
            output_dim=3,
            num_nodes=5,
            d_model=12,
            num_layers=2,
            coordinate_dim=3,
            dropout=0.0,
        ).eval()
        self.x = torch.randn(2, 5, 4)
        self.coordinates = torch.randn(2, 5, 3)
        self.adjacency = torch.ones(2, 5, 5)

    def test_e_n_invariance_and_coordinate_equivariance(self):
        output, coordinates = self.model.forward_geometric(self.x, self.coordinates, self.adjacency)
        orthogonal, _ = torch.linalg.qr(torch.randn(3, 3))
        orthogonal[:, 0] *= -1  # Include a reflection, not only SO(3) rotations.
        translation = torch.randn(1, 1, 3)
        transformed_input = self.coordinates @ orthogonal + translation
        transformed_output, transformed_coordinates = self.model.forward_geometric(
            self.x, transformed_input, self.adjacency
        )

        self.assertTrue(torch.allclose(output, transformed_output, atol=2e-5, rtol=1e-5))
        expected_coordinates = coordinates @ orthogonal + translation
        self.assertTrue(torch.allclose(expected_coordinates, transformed_coordinates, atol=2e-5, rtol=1e-5))

    def test_node_permutation_equivariance(self):
        permutation = torch.tensor([2, 4, 0, 3, 1])
        output, coordinates = self.model.forward_geometric(self.x, self.coordinates, self.adjacency)
        permuted_adjacency = self.adjacency[:, permutation][:, :, permutation]
        permuted_output, permuted_coordinates = self.model.forward_geometric(
            self.x[:, permutation], self.coordinates[:, permutation], permuted_adjacency
        )
        self.assertTrue(torch.allclose(permuted_output, output[:, permutation], atol=1e-6, rtol=1e-5))
        self.assertTrue(torch.allclose(permuted_coordinates, coordinates[:, permutation], atol=1e-6, rtol=1e-5))

    def test_geometric_batch_predict_mask_and_graph_pooling(self):
        model = EGNN(
            input_dim=4,
            output_dim=2,
            num_nodes=5,
            d_model=8,
            num_layers=1,
            task_level="graph",
            dropout=0.0,
        ).eval()
        node_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
        batch = model.make_geometric_batch(
            self.x,
            self.coordinates,
            adjacency=self.adjacency,
            node_mask=node_mask,
        )
        output = model.predict(batch)
        node_output, updated = model.forward_geometric(
            self.x,
            self.coordinates,
            self.adjacency,
            node_mask=node_mask,
        )
        self.assertIsInstance(batch, GeometricGraphBatch)
        self.assertEqual(output.shape, (2, 2))
        self.assertEqual(node_output.shape, (2, 2))
        self.assertTrue(torch.equal(updated[0, 3:], self.coordinates[0, 3:]))

    def test_edge_attributes_change_output(self):
        model = EGNN(
            input_dim=4,
            output_dim=2,
            num_nodes=5,
            d_model=8,
            num_layers=1,
            edge_dim=2,
            dropout=0.0,
        ).eval()
        zeros = torch.zeros(2, 5, 5, 2)
        ones = torch.ones(2, 5, 5, 2)
        first = model(self.x, self.coordinates, self.adjacency, edge_attr=zeros)
        second = model(self.x, self.coordinates, self.adjacency, edge_attr=ones)
        self.assertFalse(torch.allclose(first, second))

    def test_registry_top_level_and_validation(self):
        self.assertIs(tl.EGNN, EGNN)
        info = tl.get_model_info("EGNN", domain="graph")
        self.assertIn("e_n_equivariance", info["supports"])
        created = tl.create_model("EGNN", domain="graph", input_dim=3, output_dim=2, num_nodes=4)
        self.assertIsInstance(created, EGNN)
        with self.assertRaises(TypeError):
            self.model.predict(self.model.make_graph_batch(self.x, adjacency=self.adjacency))
        with self.assertRaises(ValueError):
            self.model(self.x, torch.randn(2, 5, 2), self.adjacency)


if __name__ == "__main__":
    unittest.main()
