"""General graph domain base interfaces."""

from torch_like.models.graph.gat import GAT
from torch_like.models.graph.common import GraphBatch, GraphConfig, GraphModel
from torch_like.models.graph.gcn import GCN
from torch_like.models.graph.gnan import GNAN
from torch_like.models.graph.gin import GIN
from torch_like.models.graph.graph_transformer import GraphTransformer
from torch_like.models.graph.graphsage import GraphSAGE
from torch_like.models.graph.spike_graphormer import SpikeGraphormer
from torch_like.models.registry import register_model_catalog

model_zoo = {
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
    "GCN",
    "GAT",
    "GNAN",
    "GraphSAGE",
    "GIN",
    "GraphTransformer",
    "SpikeGraphormer",
    "model_zoo",
]
