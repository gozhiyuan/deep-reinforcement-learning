from typing import Sequence, Callable, Tuple, Optional

import torch
from torch import nn

import numpy as np

from infrastructure import pytorch_util as ptu


class DQNAgent(nn.Module):
    def __init__(
        self,
        observation_shape: Sequence[int],
        num_actions: int,
        make_critic: Callable[[Tuple[int, ...], int], nn.Module],
        make_optimizer: Callable[[torch.nn.ParameterList], torch.optim.Optimizer],
        make_lr_schedule: Callable[
            [torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler
        ],
        discount: float,
        target_update_period: int,
        use_double_q: bool = False,
        clip_grad_norm: Optional[float] = None,
    ):
        super().__init__()

        self.critic = make_critic(observation_shape, num_actions)
        self.target_critic = make_critic(observation_shape, num_actions)
        self.critic_optimizer = make_optimizer(self.critic.parameters())
        self.lr_scheduler = make_lr_schedule(self.critic_optimizer)

        self.observation_shape = observation_shape
        self.num_actions = num_actions
        self.discount = discount
        self.target_update_period = target_update_period
        self.clip_grad_norm = clip_grad_norm
        self.use_double_q = use_double_q

        self.critic_loss = nn.MSELoss()

        self.update_target_critic()

    def get_action(self, observation: np.ndarray, epsilon: float = 0.0) -> int:
        """
        Epsilon-greedy action selection (default epsilon=0 for deterministic/greedy policy).
        """
        if np.random.random() < epsilon:
            return int(np.random.randint(self.num_actions))

        observation = ptu.from_numpy(np.asarray(observation))[None]

        with torch.no_grad():
            qa_values = self.critic(observation)
            action = torch.argmax(qa_values, dim=1)

        return ptu.to_numpy(action).squeeze(0).item()

    def update_critic(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
    ) -> dict:
        """Update the DQN critic, and return stats for logging."""
        # Batch shapes:
        #   obs:      (B, *observation_shape), e.g. (32, 4, 84, 84)
        #   action:   (B,), integer action taken in each sampled transition
        #   reward:   (B,), scalar reward from env.step(action)
        #   next_obs: (B, *observation_shape)
        #   done:     (B,), True only for real terminal transitions
        (batch_size,) = reward.shape

        # Compute TD targets without backpropagating through the target network.
        # For each transition:
        #   target = reward + discount * (1 - done) * Q_target(next_obs, best_next_action)
        with torch.no_grad():
            # Q-values for all actions at the next states: shape (B, num_actions).
            next_qa_values = self.target_critic(next_obs)

            if self.use_double_q:
                # Double DQN: choose the best next action with the current critic,
                # but evaluate that action with the target critic below.
                next_action = torch.argmax(self.critic(next_obs), dim=1)
            else:
                # Standard DQN: choose and evaluate the best action with target critic.
                next_action = torch.argmax(next_qa_values, dim=1)

            # Select Q_target(next_obs_i, next_action_i) from each row.
            # next_action[:, None] has shape (B, 1), gather returns (B, 1).
            next_q_values = torch.gather(
                next_qa_values, dim=1, index=next_action[:, None]
            ).squeeze(1)
            assert next_q_values.shape == (batch_size,), next_q_values.shape

            # If done=True, there is no future value, so only reward remains.
            target_values = reward + self.discount * (1 - done.float()) * next_q_values
            assert target_values.shape == (batch_size,), target_values.shape

        # Current critic predictions for all actions at sampled states:
        # shape (B, num_actions).
        qa_values = self.critic(obs)
        # Keep only Q_current(obs_i, action_i), the value of the action that was
        # actually taken in the replay transition.
        q_values = torch.gather(
            qa_values, dim=1, index=action.long()[:, None]
        ).squeeze(1)

        # Fit current Q-values toward the TD targets.
        loss = self.critic_loss(q_values, target_values)

        # One optimizer step on the current critic network.
        self.critic_optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(
            self.critic.parameters(), self.clip_grad_norm or float("inf")
        )
        self.critic_optimizer.step()

        self.lr_scheduler.step()

        # Return scalar training diagnostics; these are logged by run_dqn.py.
        return {
            "critic_loss": loss.item(),
            "q_values": q_values.mean().item(),
            "target_values": target_values.mean().item(),
            "grad_norm": grad_norm.item(),
        }

    def update_target_critic(self):
        self.target_critic.load_state_dict(self.critic.state_dict())

    def update(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
        step: int,
    ) -> dict:
        """
        Update the DQN agent, including both the critic and target.
        """
        # Update the online/current critic every training step.
        critic_stats = self.update_critic(obs, action, reward, next_obs, done)
        if step % self.target_update_period == 0:
            # Periodically sync the slower target critic to the current critic.
            # This stabilizes bootstrapped TD targets.
            self.update_target_critic()

        return critic_stats
