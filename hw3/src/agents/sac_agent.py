from typing import Callable, Optional, Sequence, Tuple
import copy

import torch
from torch import nn
import numpy as np

from infrastructure import pytorch_util as ptu


class SoftActorCritic(nn.Module):
    def __init__(
        self,
        observation_shape: Sequence[int],
        action_dim: int,
        make_actor: Callable[[Tuple[int, ...], int], nn.Module],
        make_actor_optimizer: Callable[[torch.nn.ParameterList], torch.optim.Optimizer],
        make_actor_schedule: Callable[
            [torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler
        ],
        make_critic: Callable[[Tuple[int, ...], int], nn.Module],
        make_critic_optimizer: Callable[
            [torch.nn.ParameterList], torch.optim.Optimizer
        ],
        make_critic_schedule: Callable[
            [torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler
        ],
        discount: float,
        target_update_period: Optional[int] = None,
        soft_target_update_rate: Optional[float] = None,
        # Actor-critic configuration
        num_critic_updates: int = 1,
        # Settings for multiple critics
        num_critic_networks: int = 1,
        target_critic_backup_type: str = "mean",  # One of "min" or "mean"
        # Soft actor-critic
        use_entropy_bonus: bool = False,
        temperature: float = 0.0,
        backup_entropy: bool = True,
        # Automatic temperature tuning (Section 3.5)
        auto_tune_temperature: bool = False,
        alpha_learning_rate: float = 3e-4,
    ):
        super().__init__()

        assert target_critic_backup_type in [
            "min",
            "mean",
        ], f"{target_critic_backup_type} is not a valid target critic backup type"

        assert (
            target_update_period is not None or soft_target_update_rate is not None
        ), "Must specify either target_update_period or soft_target_update_rate"

        self.actor = make_actor(observation_shape, action_dim)
        self.actor_optimizer = make_actor_optimizer(self.actor.parameters())
        self.actor_lr_scheduler = make_actor_schedule(self.actor_optimizer)

        self.critics = nn.ModuleList(
            [
                make_critic(observation_shape, action_dim)
                for _ in range(num_critic_networks)
            ]
        )

        self.critic_optimizer = make_critic_optimizer(self.critics.parameters())
        self.critic_lr_scheduler = make_critic_schedule(self.critic_optimizer)
        self.target_critics = nn.ModuleList(
            [
                make_critic(observation_shape, action_dim)
                for _ in range(num_critic_networks)
            ]
        )

        self.observation_shape = observation_shape
        self.action_dim = action_dim
        self.discount = discount
        self.target_update_period = target_update_period
        self.target_critic_backup_type = target_critic_backup_type
        self.num_critic_networks = num_critic_networks
        self.use_entropy_bonus = use_entropy_bonus
        self.temperature = temperature
        self.num_critic_updates = num_critic_updates
        self.soft_target_update_rate = soft_target_update_rate
        self.backup_entropy = backup_entropy

        # Automatic temperature tuning (Section 3.5)
        self.auto_tune_temperature = auto_tune_temperature
        if self.auto_tune_temperature:
            # TODO(Section 3.5): Initialize log_alpha, alpha_optimizer, and target_entropy
            # Hint: Initialize log_alpha to log(temperature) so alpha starts at the given temperature
            # Store log(alpha), not alpha, so exp(log_alpha) is always positive.
            self.log_alpha = nn.Parameter(
                torch.tensor(
                    np.log(temperature),
                    dtype=torch.float32,
                    device=ptu.device,
                )
            )
            self.alpha_optimizer = torch.optim.Adam(
                [self.log_alpha],
                lr=alpha_learning_rate,
            )
            self.target_entropy = -float(action_dim)
            # ENDTODO

        self.critic_loss = nn.MSELoss()

        self.update_target_critic()

    def get_temperature(self) -> float:
        """
        Get the current temperature value (either fixed or learned).
        """
        if self.auto_tune_temperature:
            # TODO(Section 3.5): Return the current learned temperature
            return self.log_alpha.exp().detach()
            # ENDTODO
        else:
            return self.temperature

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        """
        Compute the action for a given observation.
        """
        with torch.no_grad():
            # Rollout action selection does not train the actor, so skip
            # autograd graph construction to save memory and compute.
            #
            # Env observations arrive unbatched, e.g. (4,) for
            # InvertedPendulum-v4. The actor expects a batch, so [None]
            # changes this to (1, obs_dim).
            observation = ptu.from_numpy(observation)[None]

            # The SAC actor returns a distribution over continuous actions,
            # not a single deterministic action. For a batch of one state, the
            # sampled action has shape (1, action_dim), e.g. (1, 1) for
            # InvertedPendulum-v4 or (1, 3) for Hopper-v4.
            action_distribution: torch.distributions.Distribution = self.actor(observation)
            action: torch.Tensor = action_distribution.sample()

            assert action.shape == (1, self.action_dim), action.shape
            # Remove the batch dimension before passing the action to env.step:
            # (1, action_dim) -> (action_dim,).
            return ptu.to_numpy(action).squeeze(0)

    def critic(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Compute the (ensembled) Q-values for the given state-action pair.
        """
        return torch.stack([critic(obs, action) for critic in self.critics], dim=0)

    def target_critic(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Compute the (ensembled) target Q-values for the given state-action pair.
        """
        return torch.stack(
            [critic(obs, action) for critic in self.target_critics], dim=0
        )

    def q_backup_strategy(self, next_qs: torch.Tensor) -> torch.Tensor:
        """
        Combine critic estimates.

        Input example with two critics and batch size 3:
            next_qs.shape == (2, 3)
            next_qs[0] = Q1 values for the 3 samples
            next_qs[1] = Q2 values for the 3 samples

        "mean" averages critics. "min" uses clipped double-Q: min(Q1, Q2).

        Parameters:
            next_qs (torch.Tensor): Q-values of shape (num_critics, batch_size).
        Returns:
            torch.Tensor: Combined values expanded to (num_critics, batch_size).
        """

        assert (
            next_qs.ndim == 2
        ), f"next_qs should have shape (num_critics, batch_size) but got {next_qs.shape}"
        num_critic_networks, batch_size = next_qs.shape
        assert num_critic_networks == self.num_critic_networks

        # TODO(Section 3.6): Implement the "min" backup strategy (clipped double-Q).
        if self.target_critic_backup_type == "mean":
            # (num_critics, batch_size) -> (batch_size,)
            next_qs = next_qs.mean(dim=0)
        elif self.target_critic_backup_type == "min":
            # (num_critics, batch_size) -> (batch_size,)
            next_qs = next_qs.min(dim=0).values
        else:
            raise ValueError(
                f"Invalid critic backup strategy {self.target_critic_backup_type}"
            )
        # ENDTODO

        # Train each critic against the same combined target:
        #   (batch_size,) -> (num_critics, batch_size)
        if next_qs.shape == (batch_size,):
            next_qs = next_qs[None].expand((self.num_critic_networks, batch_size)).contiguous()

        assert next_qs.shape == (
            self.num_critic_networks,
            batch_size,
        ), next_qs.shape
        return next_qs

    def update_critic(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
    ):
        """
        Update the critic networks by computing target values and minimizing Bellman error.

        Batch example, InvertedPendulum with batch_size=128:
            obs:      (128, 4)
            action:   (128, 1)
            reward:   (128,)
            next_obs: (128, 4)
            done:     (128,)

        Target:
            reward + discount * (Q_target(next_obs, actor(next_obs)) + entropy_bonus)
        """
        (batch_size,) = reward.shape

        # Build bootstrap targets. no_grad() keeps actor/target critic fixed for
        # this critic update.
        with torch.no_grad():
            # TODO(Section 3.2): Sample from the actor and compute next Q-values
            # actor(next_obs) -> distribution over actions.
            # sample() -> next_action, shape (batch_size, action_dim).
            next_action_distribution = self.actor(next_obs)
            next_action = next_action_distribution.sample()
            assert next_action.shape == (batch_size, self.action_dim), next_action.shape
            # target_critic output shape: (num_critics, batch_size).
            next_qs = self.target_critic(next_obs, next_action)
            # ENDTODO

            if self.use_entropy_bonus and self.backup_entropy:
                # TODO(Section 3.3): Add entropy bonus to the target values for SAC
                # entropy shape: (batch_size,). temperature scales exploration.
                # next_qs stays (num_critics, batch_size) via [None] broadcast.
                next_action_entropy = self.entropy(next_action_distribution)
                assert next_action_entropy.shape == (batch_size,), next_action_entropy.shape
                next_qs = next_qs + self.get_temperature() * next_action_entropy[None]
                # ENDTODO

            # Handle Q-values from multiple different target critic networks (if necessary)
            next_qs = self.q_backup_strategy(next_qs)

            assert next_qs.shape == (
                self.num_critic_networks,
                batch_size,
            ), next_qs.shape

            # TODO(Section 3.2): Compute the target Q-value
            # reward[None]: (1, batch_size), next_qs: (num_critics, batch_size).
            # done=True removes future value for terminal transitions.
            target_values = reward[None] + self.discount * (1 - done.float())[None] * next_qs
            # ENDTODO
            assert target_values.shape == (
                self.num_critic_networks,
                batch_size,
            ), target_values.shape

        # TODO(Section 3.2): Update the critic
        # Predict Q-values
        # Current critic evaluates replay action:
        #   critic(obs, action) -> (num_critics, batch_size)
        q_values = self.critic(obs, action)
        assert q_values.shape == (self.num_critic_networks, batch_size), q_values.shape

        # Compute loss
        # MSE between current Q and bootstrap target.
        loss = self.critic_loss(q_values, target_values)
        # ENDTODO

        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

        return {
            "critic_loss": loss.item(),
            "q_values": q_values.mean().item(),
            "target_values": target_values.mean().item(),
        }

    def entropy(self, action_distribution: torch.distributions.Distribution):
        """
        Compute the (approximate) entropy of the action distribution for each batch element.
        """

        # TODO(Section 3.3): Compute the entropy of the action distribution.
        # Note: Think about whether to use .rsample() or .sample() here...
        # rsample(): action = mean + std * epsilon, keeps gradients.
        # -log_prob(action) is a one-sample entropy estimate, shape (batch_size,).
        action = action_distribution.rsample()
        entropy = -action_distribution.log_prob(action)
        return entropy
        # ENDTODO

    def actor_loss_reparametrize(self, obs: torch.Tensor):
        batch_size = obs.shape[0]

        # obs -> action distribution. Example Hopper:
        #   obs: (batch_size, 11), action sample: (batch_size, 3)
        action_distribution: torch.distributions.Distribution = self.actor(obs)

        # TODO(Section 3.4): Sample actions using reparameterization (replace the placeholder below)
        # Note: Think about whether to use .rsample() or .sample() here, and why...
        # rsample() lets actor gradients flow through the sampled action.
        action = action_distribution.rsample()
        assert action.shape == (batch_size, self.action_dim), action.shape
        # ENDTODO

        # TODO(Section 3.4): Compute Q-values for the sampled state-action pair (replace the placeholder below)
        # Score the actor's actions:
        #   q_values: (num_critics, batch_size)
        q_values = self.critic(obs, action)
        assert q_values.shape == (self.num_critic_networks, batch_size), q_values.shape
        # ENDTODO

        # Compute log probabilities for alpha update (Section 3.5)
        # log pi(action | obs), shape (batch_size,).
        log_prob = action_distribution.log_prob(action)

        # TODO(Section 3.4): Compute the actor loss (replace the placeholder below)
        # Minimize -Q, i.e. maximize critic value of actor actions.
        loss = -self.q_backup_strategy(q_values).mean()
        # ENDTODO

        return loss, torch.mean(self.entropy(action_distribution)), log_prob

    def update_actor(self, obs: torch.Tensor):
        """
        Update the actor by one gradient step using reparameterization.
        """
        loss, entropy, log_prob = self.actor_loss_reparametrize(obs)

        # TODO(Section 3.3): Add the entropy bonus to the actor loss: loss -= [your entropy bonus here]
        # Final SAC actor loss: -Q - temperature * entropy.
        if self.use_entropy_bonus:
            loss = loss - self.get_temperature() * entropy
        # ENDTODO

        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

        return {
            "actor_loss": loss.item(),
            "entropy": entropy.item(),
            "log_prob": log_prob.detach(),  # Detach for alpha update
        }

    def update_alpha(self, log_prob: torch.Tensor):
        """
        Update temperature parameter using dual gradient descent (Section 3.5).

        The goal is to maintain a minimum entropy level (target_entropy).
        If current entropy is below target, alpha should increase to encourage exploration.
        If current entropy is above target, alpha should decrease.

        Args:
            log_prob: Log probability of actions from current policy, shape (batch_size,)
                     Should be detached from actor gradients.

        Returns:
            dict: Dictionary with alpha and alpha_loss for logging
        """
        if not self.auto_tune_temperature:
            return {}

        # TODO(Section 3.5): Implement dual gradient descent for temperature tuning
        alpha = self.log_alpha.exp()
        # log_prob + target_entropy has shape (batch_size,). Detach it so the
        # alpha update only changes log_alpha, not the actor.
        alpha_loss = -(alpha * (log_prob.detach() + self.target_entropy)).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        alpha = self.log_alpha.exp()
        return {
            "alpha": alpha.item(),
            "alpha_loss": alpha_loss.item(),
        }
        # ENDTODO

    def update_target_critic(self):
        # Hard target update: copy current critic weights exactly.
        self.soft_update_target_critic(1.0)

    def soft_update_target_critic(self, tau):
        # target <- (1 - tau) * target + tau * current
        for target_critic, critic in zip(self.target_critics, self.critics):
            for target_param, param in zip(
                target_critic.parameters(), critic.parameters()
            ):
                target_param.data.copy_(
                    target_param.data * (1.0 - tau) + param.data * tau
                )

    def update(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
        step: int,
    ):
        """
        Update the actor and critic networks.
        """

        critic_infos = []
        # TODO(Section 3.2): Update the critic for num_critic_updates steps
        # Reuse the same replay minibatch for each critic update.
        for _ in range(self.num_critic_updates):
            info = self.update_critic(
                observations,
                actions,
                rewards,
                next_observations,
                dones,
            )
            critic_infos.append(info)
        # ENDTODO

        # TODO(Section 3.3): Enable the actor update (once you have implemented entropy)
        # Actor uses replay observations, but samples new actions from itself.
        actor_info = self.update_actor(observations)
        # ENDTODO

        # Update alpha (temperature) using dual gradient descent (Section 3.5)
        if self.auto_tune_temperature:
            alpha_info = self.update_alpha(actor_info["log_prob"])
        else:
            alpha_info = {}

        # TODO(Section 3.2): Perform either hard or soft target updates.
        # Relevant variables:
        #  - step
        #  - self.target_update_period (None when using soft updates)
        #  - self.soft_target_update_rate (None when using hard updates)
        # Soft: update every step. Hard: copy every target_update_period steps.
        if self.soft_target_update_rate is not None:
            self.soft_update_target_critic(self.soft_target_update_rate)
        elif step % self.target_update_period == 0:
            self.update_target_critic()
        # ENDTODO

        # Average the critic info over all of the steps
        critic_info = {
            k: np.mean([info[k] for info in critic_infos]) for k in critic_infos[0]
        }

        # Deal with LR scheduling
        self.actor_lr_scheduler.step()
        self.critic_lr_scheduler.step()

        # Remove log_prob from actor_info before returning (only needed for alpha update)
        if "log_prob" in actor_info:
            del actor_info["log_prob"]

        result = {
            **actor_info,
            **critic_info,
            **alpha_info,
            "actor_lr": self.actor_lr_scheduler.get_last_lr()[0],
            "critic_lr": self.critic_lr_scheduler.get_last_lr()[0],
        }

        # Always log temperature (whether fixed or learned)
        temperature = self.get_temperature()
        if isinstance(temperature, torch.Tensor):
            temperature = temperature.item()
        result["temperature"] = temperature

        return result
