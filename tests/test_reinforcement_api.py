import unittest

import torch

import torch_like as tl
from torch_like.models import create_model


class ReinforcementApiTests(unittest.TestCase):
    def test_dreamerv3_world_model_imagination_and_registry_api(self):
        model = create_model(
            "DreamerV3",
            domain="reinforcement",
            state_dim=5,
            action_dim=2,
            d_model=16,
            hidden_layers=1,
            deter_dim=12,
            stochastic_dim=4,
            classes=4,
            imagination_horizon=3,
            free_nats=0.1,
        )
        states = torch.randn(3, 4, 5)
        actions = torch.randn(3, 4, 2).tanh()
        rewards = torch.randn(3, 4)
        dones = torch.zeros(3, 4)
        dones[:, -1] = 1
        batch = model.make_reinforcement_batch(states, actions=actions, rewards=rewards, dones=dones)
        outputs = model(states, actions)
        imagined = model.imagine(outputs["posterior"], horizon=3)
        losses = model.actor_critic_loss(outputs["posterior"], horizon=3)
        world_loss = model.loss_step(batch)
        (world_loss + losses["actor_loss"] + losses["critic_loss"]).backward()
        self.assertEqual(outputs["posterior"]["logits"].shape, (3, 4, 4, 4))
        self.assertEqual(outputs["reconstructed_states"].shape, states.shape)
        self.assertEqual(outputs["predicted_rewards"].shape, rewards.shape)
        self.assertEqual(imagined["features"].shape, (3, 3, 28))
        self.assertEqual(imagined["actions"].shape, (3, 3, 2))
        self.assertEqual(losses["imagined_returns"].shape, (3, 3))
        self.assertEqual(world_loss.dim(), 0)
        self.assertIsNotNone(model.rssm.recurrent.weight_hh.grad)
        self.assertIsNotNone(model.actor[-1].weight.grad)

    def test_dqn_q_values_actions_and_td_loss(self):
        model = tl.DQN(state_dim=5, action_dim=3, d_model=16, hidden_layers=1)
        states = torch.randn(4, 5)
        next_states = torch.randn(4, 5)
        actions = torch.tensor([0, 1, 2, 1])
        rewards = torch.randn(4)
        dones = torch.tensor([0.0, 1.0, 0.0, 0.0])
        batch = model.make_reinforcement_batch(
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            dones=dones,
        )
        q_values = model.q_values(states)
        greedy = model.greedy_action(states)
        loss = model.loss_step(batch)
        loss.backward()
        self.assertEqual(q_values.shape, (4, 3))
        self.assertEqual(greedy.shape, (4,))
        self.assertEqual(loss.dim(), 0)

    def test_dqn_from_registry(self):
        model = create_model("DQN", domain="reinforcement", state_dim=5, action_dim=3, d_model=16, hidden_layers=1)
        self.assertEqual(model(torch.randn(2, 5)).shape, (2, 3))

    def test_actor_critic_policy_value_and_loss(self):
        model = tl.ActorCritic(state_dim=5, action_dim=3, d_model=16, hidden_layers=1)
        states = torch.randn(4, 5)
        actions, log_probs, values = model.act(states)
        deterministic_actions, _, _ = model.act(states, deterministic=True)
        batch = model.make_reinforcement_batch(
            states=states,
            actions=actions,
            returns=torch.randn(4),
            advantages=torch.randn(4),
        )
        loss = model.loss_step(batch)
        loss.backward()
        self.assertEqual(model.policy_logits(states).shape, (4, 3))
        self.assertEqual(model.value(states).shape, (4,))
        self.assertEqual(actions.shape, (4,))
        self.assertEqual(deterministic_actions.shape, (4,))
        self.assertEqual(log_probs.shape, (4,))
        self.assertEqual(values.shape, (4,))
        self.assertEqual(loss.dim(), 0)

    def test_ppo_actor_critic_clip_loss(self):
        model = tl.PPOActorCritic(state_dim=5, action_dim=3, d_model=16, hidden_layers=1)
        states = torch.randn(4, 5)
        actions, old_log_probs, _ = model.act(states)
        batch = model.make_reinforcement_batch(
            states=states,
            actions=actions,
            old_log_probs=old_log_probs.detach(),
            returns=torch.randn(4),
            advantages=torch.randn(4),
        )
        loss = model.loss_step(batch)
        loss.backward()
        self.assertEqual(loss.dim(), 0)

    def test_ppo_from_registry(self):
        model = create_model(
            "PPOActorCritic",
            domain="reinforcement",
            state_dim=5,
            action_dim=3,
            d_model=16,
            hidden_layers=1,
        )
        logits, values = model(torch.randn(2, 5))
        self.assertEqual(logits.shape, (2, 3))
        self.assertEqual(values.shape, (2,))

    def test_sac_actor_critic_continuous_control_api(self):
        model = tl.SACActorCritic(state_dim=5, action_dim=2, d_model=16, hidden_layers=1)
        states = torch.randn(4, 5)
        next_states = torch.randn(4, 5)
        actions, log_probs = model.sample_action(states)
        deterministic_actions, deterministic_log_probs = model(states, deterministic=True)
        q1, q2 = model.q_values(states, actions)
        batch = model.make_reinforcement_batch(
            states=states,
            actions=actions.detach(),
            rewards=torch.randn(4),
            next_states=next_states,
            dones=torch.tensor([0.0, 1.0, 0.0, 0.0]),
        )
        critic_loss = model.loss_step(batch)
        actor_loss = model.actor_loss(states)
        (critic_loss + actor_loss).backward()
        self.assertEqual(actions.shape, (4, 2))
        self.assertEqual(log_probs.shape, (4,))
        self.assertEqual(deterministic_actions.shape, (4, 2))
        self.assertEqual(deterministic_log_probs.shape, (4,))
        self.assertEqual(q1.shape, (4,))
        self.assertEqual(q2.shape, (4,))
        self.assertEqual(critic_loss.dim(), 0)
        self.assertEqual(actor_loss.dim(), 0)
        self.assertEqual(model.gaussian_entropy_target(), -2.0)

    def test_sac_from_registry(self):
        model = create_model(
            "SACActorCritic",
            domain="reinforcement",
            state_dim=5,
            action_dim=2,
            d_model=16,
            hidden_layers=1,
        )
        actions, log_probs = model(torch.randn(2, 5))
        self.assertEqual(actions.shape, (2, 2))
        self.assertEqual(log_probs.shape, (2,))

    def test_td3_actor_critic_continuous_control_api(self):
        model = tl.TD3ActorCritic(state_dim=5, action_dim=2, d_model=16, hidden_layers=1)
        states = torch.randn(4, 5)
        next_states = torch.randn(4, 5)
        actions = model.policy(states)
        target_actions = model.target_actions(next_states, noise=torch.zeros(4, 2))
        q1, q2 = model.q_values(states, actions)
        batch = model.make_reinforcement_batch(
            states=states,
            actions=actions.detach(),
            rewards=torch.randn(4),
            next_states=next_states,
            dones=torch.tensor([0.0, 1.0, 0.0, 0.0]),
        )
        critic_loss = model.loss_step(batch)
        actor_loss = model.actor_loss(states)
        (critic_loss + actor_loss).backward()
        self.assertEqual(actions.shape, (4, 2))
        self.assertEqual(target_actions.shape, (4, 2))
        self.assertLessEqual(target_actions.abs().max().item(), 1.0)
        self.assertEqual(q1.shape, (4,))
        self.assertEqual(q2.shape, (4,))
        self.assertEqual(critic_loss.dim(), 0)
        self.assertEqual(actor_loss.dim(), 0)

    def test_td3_from_registry(self):
        model = create_model(
            "TD3ActorCritic",
            domain="reinforcement",
            state_dim=5,
            action_dim=2,
            d_model=16,
            hidden_layers=1,
        )
        actions = model(torch.randn(2, 5))
        self.assertEqual(actions.shape, (2, 2))

    def test_decision_transformer_trajectory_api(self):
        model = tl.DecisionTransformer(
            state_dim=5,
            action_dim=2,
            d_model=12,
            n_heads=2,
            n_layers=1,
            d_ff=24,
            max_seq_len=4,
            max_timestep=32,
        )
        states = torch.randn(3, 4, 5)
        actions = torch.randn(3, 4, 2).clamp(-1, 1)
        returns_to_go = torch.randn(3, 4, 1)
        timesteps = torch.arange(4).unsqueeze(0).expand(3, 4)
        attention_mask = torch.tensor(
            [
                [1, 1, 1, 1],
                [1, 1, 1, 0],
                [1, 1, 0, 0],
            ]
        )
        batch = model.make_reinforcement_batch(
            states=states,
            actions=actions,
            returns_to_go=returns_to_go,
            timesteps=timesteps,
            attention_mask=attention_mask,
        )
        outputs = model(states, actions, returns_to_go, timesteps=timesteps, attention_mask=attention_mask)
        next_actions = model.predict_action(
            states,
            actions,
            returns_to_go,
            timesteps=timesteps,
            attention_mask=attention_mask,
        )
        loss = model.loss_step(batch)
        loss.backward()
        self.assertEqual(outputs["state_preds"].shape, (3, 4, 5))
        self.assertEqual(outputs["action_preds"].shape, (3, 4, 2))
        self.assertEqual(outputs["return_preds"].shape, (3, 4, 1))
        self.assertEqual(next_actions.shape, (3, 2))
        self.assertLessEqual(next_actions.abs().max().item(), 1.0)
        self.assertEqual(loss.dim(), 0)

    def test_decision_transformer_from_registry(self):
        model = create_model(
            "DecisionTransformer",
            domain="reinforcement",
            state_dim=5,
            action_dim=2,
            d_model=8,
            n_heads=2,
            n_layers=1,
            d_ff=16,
            max_seq_len=3,
        )
        outputs = model(
            torch.randn(2, 3, 5),
            torch.randn(2, 3, 2),
            torch.randn(2, 3, 1),
        )
        self.assertEqual(outputs["action_preds"].shape, (2, 3, 2))


if __name__ == "__main__":
    unittest.main()
