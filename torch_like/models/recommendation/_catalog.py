"""Recommendation domain interfaces and models."""

from torch_like.models.recommendation.bert4rec import BERT4Rec, BERT4RecConfig
from torch_like.models.recommendation.common import RecommendationBatch, RecommendationConfig, RecommendationModel
from torch_like.models.recommendation.dcn import DCN, DCNConfig
from torch_like.models.recommendation.deepfm import DeepFM, DeepFMConfig
from torch_like.models.recommendation.lightgcn import LightGCN, LightGCNConfig
from torch_like.models.recommendation.ncf import NCF, NCFConfig
from torch_like.models.recommendation.sasrec import SASRec, SASRecConfig
from torch_like.models.registry import register_model_catalog

model_zoo = {
    "NCF": {
        "cls": NCF,
        "family": "collaborative_filtering",
        "layer": "Recommendation / collaborative filtering",
        "integration_level": "paper_aligned_core",
        "supports": ["user_item_interaction", "embedding", "ranking", "implicit_feedback"],
        "paper": "Neural Collaborative Filtering (WWW 2017)",
        "description": "GMF plus MLP neural collaborative filtering model over user and item ids",
        "complexity": "O(B * embedding_dim + B * MLP) user-item interaction scoring",
    },
    "DeepFM": {
        "cls": DeepFM,
        "family": "ctr_prediction",
        "layer": "Recommendation / CTR prediction",
        "integration_level": "paper_aligned_core",
        "supports": [
            "categorical_features",
            "continuous_features",
            "factorization_machine",
            "deep_mlp",
            "ctr_prediction",
        ],
        "paper": "DeepFM: A Factorization-Machine based Neural Network for CTR Prediction (IJCAI 2017)",
        "description": "Shared embedding model combining linear terms, second-order FM interactions, and deep MLP features",
        "complexity": "O(B * fields * embedding_dim + B * MLP) CTR interaction modeling",
    },
    "DCN": {
        "cls": DCN,
        "family": "ctr_prediction",
        "layer": "Recommendation / CTR prediction",
        "integration_level": "paper_aligned_core",
        "supports": ["categorical_features", "continuous_features", "cross_network", "deep_mlp", "ctr_prediction"],
        "paper": "Deep & Cross Network for Ad Click Predictions (ADKDD 2017)",
        "description": "Explicit cross network plus deep MLP for bounded-degree feature crossing",
        "complexity": "O(B * cross_layers * feature_dim + B * MLP) explicit feature crossing",
    },
    "LightGCN": {
        "cls": LightGCN,
        "family": "graph_collaborative_filtering",
        "layer": "Recommendation / graph collaborative filtering",
        "integration_level": "paper_aligned_core",
        "supports": ["user_item_graph", "bpr_loss", "full_sort", "implicit_feedback", "ranking"],
        "paper": "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation (SIGIR 2020)",
        "description": "Linear neighborhood propagation over a normalized user-item graph for collaborative filtering",
        "complexity": "O(L * |E| * d + B * d) graph propagation plus pair scoring",
    },
    "SASRec": {
        "cls": SASRec,
        "family": "sequential_recommendation",
        "layer": "Recommendation / sequential recommendation",
        "integration_level": "paper_aligned_core",
        "supports": ["item_sequence", "causal_attention", "next_item_prediction", "candidate_scoring"],
        "paper": "Self-Attentive Sequential Recommendation (ICDM 2018)",
        "description": "Causal Transformer encoder for next-item recommendation from historical item sequences",
        "complexity": "O(B * T^2 * d_model) self-attention over item history",
    },
    "BERT4Rec": {
        "cls": BERT4Rec,
        "family": "sequential_recommendation",
        "layer": "Recommendation / sequential recommendation",
        "integration_level": "paper_aligned_core",
        "supports": ["item_sequence", "bidirectional_attention", "masked_item_prediction", "candidate_scoring"],
        "paper": "BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer (CIKM 2019)",
        "description": "Bidirectional Transformer encoder for Cloze-style masked item prediction over item sequences",
        "complexity": "O(B * T^2 * d_model) bidirectional self-attention over item history",
    },
}

register_model_catalog("recommendation", model_zoo)

__all__ = [
    "RecommendationConfig",
    "RecommendationBatch",
    "RecommendationModel",
    "NCFConfig",
    "NCF",
    "DeepFMConfig",
    "DeepFM",
    "DCNConfig",
    "DCN",
    "LightGCNConfig",
    "LightGCN",
    "SASRecConfig",
    "SASRec",
    "BERT4RecConfig",
    "BERT4Rec",
    "model_zoo",
]
