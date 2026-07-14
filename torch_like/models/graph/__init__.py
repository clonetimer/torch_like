"""General graph domain base interfaces."""

from torch_like.models.graph.common import GraphBatch, GraphConfig, GraphModel
from torch_like.models.graph.dgi import DGI, DGIConfig
from torch_like.models.graph.egnn import EGNN, EGNNConfig, EGNNLayer, GeometricGraphBatch
from torch_like.models.graph.gat import GAT
from torch_like.models.graph.gatv2 import GATv2, GATv2Config
from torch_like.models.graph.gcn import GCN
from torch_like.models.graph.gcnii import GCNII, GCNIIConfig
from torch_like.models.graph.gin import GIN
from torch_like.models.graph.gnan import GNAN
from torch_like.models.graph.gps import GPS, GPSConfig
from torch_like.models.graph.graph_transformer import GraphTransformer
from torch_like.models.graph.graphcl import GraphCL, GraphCLConfig
from torch_like.models.graph.graphmae import GraphMAE, GraphMAEConfig
from torch_like.models.graph.graphormer import Graphormer, GraphormerConfig
from torch_like.models.graph.graphsage import GraphSAGE
from torch_like.models.graph.pna import PNA, PNAConfig
from torch_like.models.graph.spike_graphormer import SpikeGraphormer
from torch_like.models.registry import register_model_catalog

model_zoo = {
    "EGNN": {
        "cls": EGNN,
        "family": "equivariant_message_passing",
        "layer": "Geometric graph learning",
        "integration_level": "paper_aligned_dense",
        "supports": ["adjacency", "node_task", "graph_task", "coordinates", "e_n_equivariance", "edge_attributes"],
        "paper": "E(n) Equivariant Graph Neural Networks (ICML 2021)",
        "description": "Geometric message passing with invariant features and E(n)-equivariant coordinate updates",
        "complexity": "O(B * N^2 * D) dense geometric message passing over N nodes",
    },
    "GPS": {
        "cls": GPS,
        "family": "graph_transformer",
        "layer": "General graph learning",
        "integration_level": "framework_oriented_dense",
        "supports": ["adjacency", "node_task", "graph_task", "attention", "local_global_mixing"],
        "paper": "Recipe for a General, Powerful, Scalable Graph Transformer (NeurIPS 2022)",
        "description": "GPS layer combining local adjacency propagation with global self-attention",
        "complexity": "O(B * N^2 * D) under the current dense GraphBatch backend",
    },
    "Graphormer": {
        "cls": Graphormer,
        "family": "graph_transformer",
        "layer": "General graph learning",
        "integration_level": "paper_aligned",
        "supports": ["adjacency", "node_task", "graph_task", "attention", "centrality", "spatial_encoding"],
        "paper": "Do Transformers Really Perform Badly for Graph Representation? (NeurIPS 2021)",
        "description": "Graph Transformer with degree centrality and shortest-path spatial attention biases",
        "complexity": "O(B * N^3 + B * H * N^2 * D) shortest paths plus dense graph attention",
    },
    "GNAN": {
        "cls": GNAN,
        "family": "graph_additive",
        "layer": "General graph learning",
        "integration_level": "framework_oriented",
        "supports": ["adjacency", "node_task", "graph_task", "interpretable", "distance_aware"],
        "paper": "The Intelligible and Effective Graph Neural Additive Networks (NeurIPS 2024), PyTorch-native framework integration",
        "description": "Distance-aware additive graph model with interpretable pairwise contribution weights",
        "complexity": "O(B * N^3 + B * L * N^2 * D) with shortest-path preprocessing and dense additive aggregation",
    },
    "PNA": {
        "cls": PNA,
        "family": "message_passing",
        "layer": "General graph learning",
        "integration_level": "framework_oriented_dense",
        "supports": ["adjacency", "node_task", "graph_task", "multi_aggregation", "degree_scaling"],
        "paper": "Principal Neighbourhood Aggregation for Graph Nets (NeurIPS 2020)",
        "description": "Message-passing GNN with multiple aggregators and degree-aware scalers",
        "complexity": "O(B * N^2 * D * aggregators * scalers) with dense adjacency",
    },
    "GCNII": {
        "cls": GCNII,
        "family": "deep_message_passing",
        "layer": "General graph learning",
        "integration_level": "paper_aligned_core",
        "supports": ["adjacency", "node_task", "graph_task", "deep_gcn", "initial_residual", "identity_mapping"],
        "paper": "Simple and Deep Graph Convolutional Networks (ICML 2020)",
        "description": "Deep GCN with initial residual and identity mapping to reduce over-smoothing",
        "complexity": "O(B * L * N^2 * D) with dense adjacency and L GCNII layers",
    },
    "DGI": {
        "cls": DGI,
        "family": "self_supervised_graph_representation",
        "layer": "General graph learning",
        "integration_level": "paper_aligned_core",
        "supports": ["adjacency", "node_task", "graph_task", "self_supervised", "mutual_information", "corruption"],
        "paper": "Deep Graph Infomax (ICLR 2019)",
        "description": "Graph encoder with global summary discrimination for self-supervised node and graph representations",
        "complexity": "O(B * L * N^2 * D) dense graph encoding plus positive/negative discrimination",
    },
    "GraphCL": {
        "cls": GraphCL,
        "family": "self_supervised_graph_representation",
        "layer": "General graph learning",
        "integration_level": "paper_aligned_core",
        "supports": [
            "adjacency",
            "graph_task",
            "self_supervised",
            "graph_contrastive_learning",
            "augmentation_invariant",
        ],
        "paper": "Graph Contrastive Learning with Augmentations (NeurIPS 2020)",
        "description": "Graph encoder with symmetric InfoNCE over two augmented graph views",
        "complexity": "O(B * L * N^2 * D) dense graph encoding plus O(B^2 * D) contrastive logits",
    },
    "GraphMAE": {
        "cls": GraphMAE,
        "family": "self_supervised_graph_representation",
        "layer": "General graph learning",
        "integration_level": "paper_aligned_core",
        "supports": ["adjacency", "node_task", "graph_task", "self_supervised", "masked_graph_autoencoding"],
        "paper": "GraphMAE: Self-Supervised Masked Graph Autoencoders (KDD 2022)",
        "description": "Masked node-feature graph autoencoder with reconstruction loss over selected nodes",
        "complexity": "O(B * L * N^2 * D) dense graph encoding plus decoder reconstruction",
    },
    "GCN": {
        "cls": GCN,
        "family": "message_passing",
        "layer": "General graph learning",
        "integration_level": "framework_oriented",
        "supports": ["adjacency", "node_task", "graph_task"],
        "paper": "Semi-Supervised Classification with Graph Convolutional Networks (ICLR 2017)",
        "description": "Baseline graph convolution network over normalized adjacency",
        "complexity": "O(B * N^2 * D) with dense adjacency over N nodes and hidden width D",
    },
    "SpikeGraphormer": {
        "cls": SpikeGraphormer,
        "family": "graph_transformer",
        "layer": "General graph learning",
        "integration_level": "framework_oriented",
        "supports": ["adjacency", "node_task", "graph_task", "attention", "spiking"],
        "paper": "SpikeGraphormer: A High-Performance Graph Transformer with Spiking Graph Attention (framework-oriented integration)",
        "description": "Dual-branch graph model with local graph propagation and spiking-style graph attention",
        "complexity": "O(B * N^2 * H * D) with adjacency-masked global attention and local graph propagation",
    },
    "GraphTransformer": {
        "cls": GraphTransformer,
        "family": "graph_transformer",
        "layer": "General graph learning",
        "integration_level": "framework_oriented",
        "supports": ["adjacency", "node_task", "graph_task", "attention"],
        "paper": "Graph Transformer baseline for torch_like graph domain (framework-oriented)",
        "description": "Dense graph self-attention baseline with optional adjacency masking",
        "complexity": "O(B * N^2 * H * D) with dense graph attention over N nodes, heads H, hidden width D",
    },
    "GAT": {
        "cls": GAT,
        "family": "graph_attention",
        "layer": "General graph learning",
        "integration_level": "framework_oriented",
        "supports": ["adjacency", "node_task", "graph_task", "attention"],
        "paper": "Graph Attention Networks (ICLR 2018)",
        "description": "Masked self-attention over graph neighborhoods",
        "complexity": "O(B * N^2 * H * D) with dense attention over N nodes, heads H, hidden width D",
    },
    "GATv2": {
        "cls": GATv2,
        "family": "graph_attention",
        "layer": "General graph learning",
        "integration_level": "framework_oriented_dense",
        "supports": ["adjacency", "node_task", "graph_task", "attention", "dynamic_attention"],
        "paper": "How Attentive are Graph Attention Networks? (ICLR 2022)",
        "description": "GATv2-style dynamic graph attention with pairwise source-destination scoring",
        "complexity": "O(B * N^2 * H * D) with dense dynamic graph attention",
    },
    "GraphSAGE": {
        "cls": GraphSAGE,
        "family": "message_passing",
        "layer": "General graph learning",
        "integration_level": "framework_oriented",
        "supports": ["adjacency", "node_task", "graph_task", "inductive"],
        "paper": "Inductive Representation Learning on Large Graphs (NeurIPS 2017)",
        "description": "Neighborhood mean aggregation with shared inductive encoder",
        "complexity": "O(B * N^2 * D) with dense adjacency aggregation over neighbors",
    },
    "GIN": {
        "cls": GIN,
        "family": "message_passing",
        "layer": "General graph learning",
        "integration_level": "framework_oriented",
        "supports": ["adjacency", "node_task", "graph_task", "weisfeiler_lehman"],
        "paper": "How Powerful are Graph Neural Networks? (ICLR 2019)",
        "description": "Sum-aggregation graph network with learnable epsilon and MLP update",
        "complexity": "O(B * N^2 * D) with dense sum aggregation and MLP updates",
    },
}

register_model_catalog("graph", model_zoo)

__all__ = [
    "GraphConfig",
    "GraphBatch",
    "GraphModel",
    "GeometricGraphBatch",
    "EGNNConfig",
    "EGNNLayer",
    "EGNN",
    "GCN",
    "DGIConfig",
    "DGI",
    "GraphCLConfig",
    "GraphCL",
    "GraphMAEConfig",
    "GraphMAE",
    "GCNIIConfig",
    "GCNII",
    "GAT",
    "GATv2Config",
    "GATv2",
    "GNAN",
    "GraphSAGE",
    "GIN",
    "GraphTransformer",
    "GraphormerConfig",
    "Graphormer",
    "GPSConfig",
    "GPS",
    "PNAConfig",
    "PNA",
    "SpikeGraphormer",
    "model_zoo",
]
