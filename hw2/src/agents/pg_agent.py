from typing import Optional, Sequence
import numpy as np
import torch

from networks.critics import ValueCritic
from networks.policies import MLPPolicyPG
from infrastructure import pytorch_util as ptu
from torch import nn


class PGAgent(nn.Module):
    def __init__(
        self,
        ob_dim: int,
        ac_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        gamma: float,
        learning_rate: float,
        use_baseline: bool,
        use_reward_to_go: bool,
        baseline_learning_rate: Optional[float],
        baseline_gradient_steps: Optional[int],
        gae_lambda: Optional[float],
        normalize_advantages: bool,
    ):
        super().__init__()

        # create the actor (policy) network
        self.actor = MLPPolicyPG(
            ac_dim, ob_dim, discrete, n_layers, layer_size, learning_rate
        )

        # create the critic (baseline) network, if needed
        if use_baseline:
            self.critic = ValueCritic(
                ob_dim, n_layers, layer_size, baseline_learning_rate
            )
            self.baseline_gradient_steps = baseline_gradient_steps
        else:
            self.critic = None

        # other agent parameters
        self.gamma = gamma
        self.use_reward_to_go = use_reward_to_go
        self.gae_lambda = gae_lambda
        self.normalize_advantages = normalize_advantages

    def update(
        self,
        obs: Sequence[np.ndarray],
        actions: Sequence[np.ndarray],
        rewards: Sequence[np.ndarray],
        terminals: Sequence[np.ndarray],
    ) -> dict:
        """The train step for PG involves updating its actor using the given observations/actions and the calculated
        qvals/advantages that come from the seen rewards.

        Each input is a list of NumPy arrays, where each array is one trajectory. The trajectories may have different
        lengths.
        """

        # step 1: calculate Q values of each (s_t, a_t) point, using rewards (r_0, ..., r_t, ..., r_T)
        # Rewards = [
        # np.array([r_0, r_1, ..., r_T0]),  # trajectory 0
        # np.array([r_0, r_1, ..., r_T1]),  # trajectory 1
        # ...
        # ] 
        q_values: Sequence[np.ndarray] = self._calculate_q_vals(rewards)

        # Example before concat, CartPole ob_dim=4:
        # obs = [array shape (3, 4), array shape (5, 4)]
        # actions = [array shape (3,), array shape (5,)]
        # rewards = [array shape (3,), array shape (5,)]
        #
        # After concat, batch_size = 3 + 5 = 8:
        # obs shape -> (8, 4)
        # actions/rewards/q_values shape -> (8,)
        obs = np.concatenate(obs)
        actions = np.concatenate(actions)
        rewards = np.concatenate(rewards)
        terminals = np.concatenate(terminals)
        q_values = np.concatenate(q_values)

        # step 2: calculate advantages from Q values
        advantages: np.ndarray = self._estimate_advantage(
            obs, rewards, q_values, terminals
        )

        # step 3: use all datapoints (s_t, a_t, adv_t) to update the PG actor/policy
        info: dict = self.actor.update(obs, actions, advantages)

        # step 4: if needed, use all datapoints (s_t, a_t, q_t) to update the PG critic/baseline
        if self.critic is not None:
            critic_info = {}
            for _ in range(self.baseline_gradient_steps):
                critic_info = self.critic.update(obs, q_values)

            info.update(critic_info)

        return info

    def _discounted_return(self, rewards: Sequence[float]) -> Sequence[float]:
        """
        Helper function which takes the rewards from one completed trajectory,
        {r_0, r_1, ..., r_t', ... r_T}, and returns one Q-value per timestep.

        In the trajectory-based policy gradient estimator, every action in the
        same trajectory is assigned the same total discounted episode return:
            Q(s_t, a_t) = sum_{t'=0}^T gamma^t' r_{t'}

        Note that all entries of the output list should be the exact same because each sum is from 0 to T (and doesn't
        involve t)!
        """
        rewards = np.array(rewards, dtype=np.float32)

        # Example: rewards=[1, 1, 1], gamma=0.9 -> discounts=[1, 0.9, 0.81].
        discounts = self.gamma ** np.arange(len(rewards))

        # Full-trajectory Monte Carlo return: 1 + 0.9 + 0.81 = 2.71.
        discounted_return = np.sum(discounts * rewards)

        # Later code needs one q_value per step, so repeat the same return:
        # [2.71, 2.71, 2.71], not just scalar 2.71.
        return np.full_like(rewards, discounted_return, dtype=np.float32)

    def _discounted_reward_to_go(self, rewards: Sequence[float]) -> Sequence[float]:
        """
        Helper function which takes a list of rewards {r_0, r_1, ..., r_t', ... r_T} and returns a list where the entry
        in each index t is sum_{t'=t}^T gamma^(t'-t) * r_{t'}.
        """
        rewards = np.array(rewards, dtype=np.float32)
        discounted_reward_to_go = np.zeros_like(rewards, dtype=np.float32)
        running_sum = 0.0

        # Example: rewards=[1, 1, 1], gamma=0.9.
        # Work backward:
        # t=2 -> 1
        # t=1 -> 1 + 0.9 * 1 = 1.9
        # t=0 -> 1 + 0.9 * 1.9 = 2.71
        # discounted_reward_to_go becomes [2.71, 1.9, 1].
        for t in reversed(range(len(rewards))):
            running_sum = rewards[t] + self.gamma * running_sum
            discounted_reward_to_go[t] = running_sum

        return discounted_reward_to_go

    def _calculate_q_vals(self, rewards: Sequence[np.ndarray]) -> Sequence[np.ndarray]:
        """Monte Carlo estimation of the Q function."""

        if not self.use_reward_to_go:
            # Case 1: in trajectory-based PG, we ignore the timestep and instead use the discounted return for the entire
            # trajectory at each point.
            # In other words: Q(s_t, a_t) = sum_{t'=0}^T gamma^t' r_{t'}
            q_values = [self._discounted_return(reward) for reward in rewards]
        else:
            # Case 2: in reward-to-go PG, we only use the rewards after timestep t to estimate the Q-value for (s_t, a_t).
            # In other words: Q(s_t, a_t) = sum_{t'=t}^T gamma^(t'-t) * r_{t'}
            q_values = [self._discounted_reward_to_go(reward) for reward in rewards]

        return q_values

    def _estimate_advantage(
        self,
        obs: np.ndarray,
        rewards: np.ndarray,
        q_values: np.ndarray,
        terminals: np.ndarray,
    ) -> np.ndarray:
        """Computes advantages by (possibly) subtracting a value baseline from the estimated Q-values.

        Operates after concatenation:
        - obs: (batch_size, ob_dim)
        - rewards, q_values, terminals: (batch_size,)
        Returns:
        - advantages: (batch_size,)
        """
        if self.critic is None:
            # No baseline: use the Monte Carlo Q-value directly as the advantage.
            # Example: q_values=[2.71, 1.9, 1.0] -> advantages=[2.71, 1.9, 1.0].
            advantages = q_values.copy()
        else:
            # values[i] is the critic's estimate V(s_i), shape (batch_size,).
            # Example: obs shape (8, 4) -> values shape (8,).
            values = ptu.to_numpy(self.critic(ptu.from_numpy(obs)))
            assert values.shape == q_values.shape

            if self.gae_lambda is None:
                # Baseline advantage: A(s_i, a_i) = Q(s_i, a_i) - V(s_i).
                # Example: q_values=[2.71, 1.9], values=[2.0, 2.1]
                # -> advantages=[0.71, -0.2].
                advantages = q_values - values
            else:
                batch_size = obs.shape[0]

                # Append a dummy V(s_{T+1})=0 so values[i + 1] is always valid.
                # Also create one extra dummy advantage for advantages[i + 1].
                values = np.append(values, [0])
                advantages = np.zeros(batch_size + 1)

                # GAE works backward and reuses the future advantage:
                # delta[i] = rewards[i] + gamma * values[i + 1] - values[i]
                # advantage[i] = delta[i] + gamma * lambda * advantage[i + 1]
                #
                # Small one-trajectory example:
                # rewards=[1, 1, 1], values=[2.0, 1.5, 0.5], terminals=[0, 0, 1]
                # gamma=0.9, lambda=0.95
                #
                # i=2: terminal -> next_nonterminal=0
                #      delta = 1 + 0 - 0.5 = 0.5
                #      adv[2] = 0.5
                # i=1: delta = 1 + 0.9*0.5 - 1.5 = -0.05
                #      adv[1] = -0.05 + 0.9*0.95*0.5 = 0.3775
                # i=0: delta = 1 + 0.9*1.5 - 2.0 = 0.35
                #      adv[0] = 0.35 + 0.9*0.95*0.3775 = 0.6727625
                # Final advantages: [0.6727625, 0.3775, 0.5].
                for i in reversed(range(batch_size)):
                    # terminals[i]=1 means this is the last step of a trajectory.
                    # Then next_nonterminal=0, so we do not use values/advantages
                    # from the next trajectory by accident.
                    next_nonterminal = 1 - terminals[i]

                    # TD error:
                    # delta = r_i + gamma * V(s_{i+1}) - V(s_i)
                    # If terminal, next_nonterminal=0 and the next-value term drops.
                    delta = (
                        rewards[i]
                        + self.gamma * values[i + 1] * next_nonterminal
                        - values[i]
                    )

                    # Recursive smoothing of future TD errors. lambda=0 uses only
                    # delta[i]; lambda near 1 uses more future deltas.
                    advantages[i] = (
                        delta
                        + self.gamma
                        * self.gae_lambda
                        * next_nonterminal
                        * advantages[i + 1]
                    )

                # remove dummy advantage
                advantages = advantages[:-1]

        if self.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages
