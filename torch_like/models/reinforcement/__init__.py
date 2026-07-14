"""Reinforcement-learning domain interfaces and models."""

from torch_like.models.registry import register_model_catalog
from torch_like.models.reinforcement.actor_critic import ActorCritic, ActorCriticConfig
from torch_like.models.reinforcement.common import ReinforcementBatch, ReinforcementConfig, ReinforcementModel
from torch_like.models.reinforcement.decision_transformer import DecisionTransformer, DecisionTransformerConfig
from torch_like.models.reinforcement.dqn import DQN, DQNConfig
from torch_like.models.reinforcement.dreamerv3 import DiscreteRSSM, DreamerV3, DreamerV3Config
from torch_like.models.reinforcement.ppo_actor_critic import PPOActorCritic, PPOActorCriticConfig
from torch_like.models.reinforcement.sac_actor_critic import SACActorCritic, SACActorCriticConfig
from torch_like.models.reinforcement.td3_actor_critic import TD3ActorCritic, TD3ActorCriticConfig

model_zoo = {
    "DreamerV3": {
        "cls": DreamerV3,
        "family": "world_model_reinforcement_learning",
        "layer": "Reinforcement learning / model-based control",
        "integration_level": "paper_aligned_core",
        "supports": ["world_model", "discrete_rssm", "latent_imagination", "continuous_actions", "actor_critic", "symlog", "kl_balancing"],
        "paper": "Mastering Diverse Domains through World Models (Nature 2025)",
        "description": "Discrete RSSM world model with imagined latent rollouts and continuous actor-critic heads",
        "complexity": "O(B * T * RSSM + B * imagination_horizon * RSSM) recurrent observation and imagination",
    },
    "DQN": {
        "cls": DQN,
        "family": "value_based_rl",
        "layer": "Reinforcement learning / value functions",
        "integration_level": "paper_aligned_core",
        "supports": ["q_values", "greedy_action", "td_loss", "discrete_actions"],
        "paper": "Human-level control through deep reinforcement learning (Nature 2015)",
        "description": "MLP Q-network exposing discrete-action Q values and one-step TD loss",
        "complexity": "O(B * hidden_layers * d_model^2) feed-forward Q estimation",
    },
    "ActorCritic": {
        "cls": ActorCritic,
        "family": "policy_gradient_rl",
        "layer": "Reinforcement learning / policy gradients",
        "integration_level": "framework_oriented",
        "supports": ["policy_logits", "value_function", "action_distribution", "actor_critic_loss"],
        "paper": "Asynchronous Methods for Deep Reinforcement Learning (ICML 2016)",
        "description": "Shared encoder with categorical policy and scalar value heads for actor-critic algorithms",
        "complexity": "O(B * hidden_layers * d_model^2) policy/value forward pass",
    },
    "PPOActorCritic": {
        "cls": PPOActorCritic,
        "family": "policy_gradient_rl",
        "layer": "Reinforcement learning / policy gradients",
        "integration_level": "paper_aligned_core",
        "supports": ["policy_logits", "value_function", "ppo_clip", "entropy_bonus", "discrete_actions"],
        "paper": "Proximal Policy Optimization Algorithms (2017)",
        "description": "Actor-critic module exposing PPO clipped surrogate objective for external rollout loops",
        "complexity": "O(B * hidden_layers * d_model^2) policy/value forward pass",
    },
    "SACActorCritic": {
        "cls": SACActorCritic,
        "family": "maximum_entropy_rl",
        "layer": "Reinforcement learning / continuous control",
        "integration_level": "paper_aligned_core",
        "supports": ["continuous_actions", "gaussian_policy", "twin_q", "entropy_regularization", "soft_q_loss"],
        "paper": "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor (ICML 2018)",
        "description": "Tanh-squashed Gaussian policy with twin Q networks and SAC actor/critic losses",
        "complexity": "O(B * hidden_layers * d_model^2) actor plus twin-Q forward passes",
    },
    "TD3ActorCritic": {
        "cls": TD3ActorCritic,
        "family": "deterministic_actor_critic_rl",
        "layer": "Reinforcement learning / continuous control",
        "integration_level": "paper_aligned_core",
        "supports": ["continuous_actions", "deterministic_policy", "twin_q", "target_policy_smoothing", "td3_loss"],
        "paper": "Addressing Function Approximation Error in Actor-Critic Methods (ICML 2018)",
        "description": "Deterministic actor with twin critics, clipped target-policy smoothing, and TD3 actor/critic losses",
        "complexity": "O(B * hidden_layers * d_model^2) deterministic actor plus twin-Q forward passes",
    },
    "DecisionTransformer": {
        "cls": DecisionTransformer,
        "family": "offline_sequence_modeling_rl",
        "layer": "Reinforcement learning / offline trajectory modeling",
        "integration_level": "paper_aligned_core",
        "supports": [
            "offline_rl",
            "return_conditioning",
            "trajectory_modeling",
            "causal_attention",
            "continuous_actions",
        ],
        "paper": "Decision Transformer: Reinforcement Learning via Sequence Modeling (NeurIPS 2021)",
        "description": "Causal Transformer over return, state, and action tokens with trajectory-conditioned action prediction",
        "complexity": "O(B * (3T)^2 * d_model) causal trajectory attention",
    },
}

register_model_catalog("reinforcement", model_zoo)

__all__ = [
    "ReinforcementConfig",
    "ReinforcementBatch",
    "ReinforcementModel",
    "DreamerV3Config",
    "DiscreteRSSM",
    "DreamerV3",
    "DQNConfig",
    "DQN",
    "ActorCriticConfig",
    "ActorCritic",
    "PPOActorCriticConfig",
    "PPOActorCritic",
    "SACActorCriticConfig",
    "SACActorCritic",
    "TD3ActorCriticConfig",
    "TD3ActorCritic",
    "DecisionTransformerConfig",
    "DecisionTransformer",
    "model_zoo",
]
