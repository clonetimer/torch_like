"""Tabular domain base interfaces and models."""

from torch_like.models.registry import register_model_catalog
from torch_like.models.tabular.common import TabularBatch, TabularConfig, TabularModel
from torch_like.models.tabular.ft_transformer import FTTransformer, FTTransformerConfig
from torch_like.models.tabular.node import NODE, NODEConfig
from torch_like.models.tabular.saint import SAINT, SAINTConfig
from torch_like.models.tabular.tab_transformer import TabTransformer, TabTransformerConfig
from torch_like.models.tabular.tabnet import TabNet, TabNetConfig

model_zoo = {
    "TabNet": {
        "cls": TabNet,
        "family": "attentive_tabular_model",
        "layer": "Tabular prediction",
        "integration_level": "paper_aligned_core",
        "supports": [
            "continuous_features",
            "categorical_features",
            "feature_masks",
            "interpretability",
            "classification",
            "regression",
        ],
        "paper": "TabNet: Attentive Interpretable Tabular Learning (AAAI 2021)",
        "description": "Sequential attentive feature-mask tabular model with sparse feature selection and explainable masks",
        "complexity": "O(B * steps * feature_dim * d_model) attentive tabular feature selection",
    },
    "NODE": {
        "cls": NODE,
        "family": "neural_tree_ensemble",
        "layer": "Tabular prediction",
        "integration_level": "paper_aligned_core",
        "supports": [
            "continuous_features",
            "categorical_features",
            "oblivious_trees",
            "differentiable_splits",
            "classification",
            "regression",
        ],
        "paper": "Neural Oblivious Decision Ensembles for Deep Learning on Tabular Data (ICLR 2020)",
        "description": "Differentiable oblivious decision tree ensemble for tabular classification and regression",
        "complexity": "O(B * trees * depth * d_model + B * trees * 2^depth * output_dim)",
    },
    "FTTransformer": {
        "cls": FTTransformer,
        "family": "tabular_transformer",
        "layer": "Tabular prediction",
        "integration_level": "paper_aligned_core",
        "supports": [
            "continuous_features",
            "categorical_features",
            "feature_tokenization",
            "attention",
            "classification",
            "regression",
        ],
        "paper": "Revisiting Deep Learning Models for Tabular Data (NeurIPS 2021)",
        "description": "Feature-tokenizer Transformer for mixed continuous/categorical tabular prediction",
        "complexity": "O(B * features^2 * d_model * layers) attention over tabular feature tokens",
    },
    "TabTransformer": {
        "cls": TabTransformer,
        "family": "tabular_transformer",
        "layer": "Tabular prediction",
        "integration_level": "paper_aligned_core",
        "supports": ["categorical_features", "contextual_embeddings", "attention", "classification", "regression"],
        "paper": "TabTransformer: Tabular Data Modeling Using Contextual Embeddings (2020)",
        "description": "Transformer contextualization over categorical feature embeddings fused with continuous features",
        "complexity": "O(B * categorical_features^2 * d_model * layers) categorical token attention",
    },
    "SAINT": {
        "cls": SAINT,
        "family": "tabular_transformer",
        "layer": "Tabular prediction",
        "integration_level": "paper_aligned_core",
        "supports": [
            "continuous_features",
            "categorical_features",
            "column_attention",
            "row_attention",
            "classification",
            "regression",
        ],
        "paper": "SAINT: Improved Neural Networks for Tabular Data via Row Attention and Contrastive Pre-Training (2021)",
        "description": "Column self-attention plus optional row attention over samples for tabular prediction",
        "complexity": "O(B * features^2 * d_model * layers + features * B^2 * d_model * layers) column and row attention",
    },
}

register_model_catalog("tabular", model_zoo)

__all__ = [
    "TabularConfig",
    "TabularBatch",
    "TabularModel",
    "TabNetConfig",
    "TabNet",
    "NODEConfig",
    "NODE",
    "FTTransformerConfig",
    "FTTransformer",
    "TabTransformerConfig",
    "TabTransformer",
    "SAINTConfig",
    "SAINT",
    "model_zoo",
]
