import itertools
from torch import nn
from torch.nn import functional as F
import torch.distributions as D
from torch import optim

import numpy as np
import torch
from torch import distributions

from infrastructure import pytorch_util as ptu


class MLPPolicy(nn.Module):
    """Base MLP policy, which can take an observation and output a distribution over actions.

    This class should implement the `forward` and `get_action` methods. The `update` method should be written in the
    subclasses, since the policy update rule differs for different algorithms.
    """

    def __init__(
        self,
        ac_dim: int,
        ob_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        learning_rate: float,
    ):
        super().__init__()

        if discrete:
            # Discrete action spaces have a fixed set of action IDs.
            # Example: CartPole has ac_dim=2, with actions 0=left and 1=right.
            # Input shape to this network:  (batch_size, ob_dim)
            # Output shape from this network: (batch_size, ac_dim)
            # The outputs are logits, not probabilities. A Categorical distribution
            # will later apply softmax internally when sampling/log-probabilities
            # are needed.
            self.logits_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            parameters = self.logits_net.parameters()
        else:
            # Continuous action spaces need one real-valued action vector.
            # Example: if ac_dim=3, each action has shape (3,).
            # Input shape to this network:  (batch_size, ob_dim)
            # Output shape from this network: (batch_size, ac_dim)
            # The network predicts the Gaussian mean for each action dimension.
            self.mean_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            # logstd is a learned parameter shared across observations.
            # Shape: (ac_dim,). After exp(logstd), it becomes the Gaussian std.
            self.logstd = nn.Parameter(
                torch.zeros(ac_dim, dtype=torch.float32, device=ptu.device)
            )
            parameters = itertools.chain([self.logstd], self.mean_net.parameters())

        self.optimizer = optim.Adam(
            parameters,
            learning_rate,
        )

        self.discrete = discrete

    @torch.no_grad()
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Takes a single observation (as a numpy array) and returns a single action (as a numpy array)."""
        # The policy network expects a batch dimension. A single observation has
        # shape (ob_dim,), so convert it to shape (1, ob_dim).
        if obs.ndim == 1:
            obs = obs[None]

        # forward(...) returns a torch Distribution object:
        # - discrete: a Categorical distribution over action IDs
        # - continuous: a Gaussian distribution over action vectors
        # Sampling gives one action per observation in the batch.
        # Before removing the batch dimension:
        # - discrete action shape: (1,)
        # - continuous action shape: (1, ac_dim)
        action_distribution = self.forward(ptu.from_numpy(obs))
        action = action_distribution.sample()

        # Remove the batch dimension before giving the action to env.step(...).
        # Returned shape:
        # - discrete: scalar action ID, like 0 or 1
        # - continuous: array with shape (ac_dim,)
        return ptu.to_numpy(action)[0]

    def forward(self, obs: torch.FloatTensor):
        """
        This function defines the forward pass of the network.  You can return anything you want, but you should be
        able to differentiate through it. For example, you can return a torch.FloatTensor. You can also return more
        flexible objects, such as a `torch.distributions.Distribution` object. It's up to you!
        """
        if self.discrete:
            # logits shape: (batch_size, ac_dim)
            # sample() shape: (batch_size,), containing integer action IDs.
            # log_prob(actions) shape: (batch_size,), one log-prob per sampled ID.
            logits = self.logits_net(obs)
            return D.Categorical(logits=logits)
        else:
            # mean shape: (batch_size, ac_dim)
            # std shape: (ac_dim,), broadcast across the batch.
            # sample() shape: (batch_size, ac_dim), containing full action vectors.
            # log_prob(actions) should still be shape (batch_size,), one log-prob
            # per full action vector. Independent sums the per-dimension Normal
            # log-probs into that one value per action vector.
            mean = self.mean_net(obs)
            std = torch.exp(self.logstd)
            return D.Independent(D.Normal(mean, std), 1)

    def update(self, obs: np.ndarray, actions: np.ndarray, *args, **kwargs) -> dict:
        """
        Performs one iteration of gradient descent on the provided batch of data. You don't need to implement this
        method in the base class, but you do need to implement it in the subclass.
        """
        raise NotImplementedError


class MLPPolicyPG(MLPPolicy):
    """Policy subclass for the policy gradient algorithm."""

    def update(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
    ) -> dict:
        """Implements the policy gradient actor update."""
        # Inputs are already concatenated across trajectories by PGAgent.update.
        # Let B = batch_size = total number of env steps in the collected batch.
        # - obs: (B, ob_dim)
        # - actions: (B,) for discrete or (B, ac_dim) for continuous
        # - advantages: (B,), one scalar weight per sampled action
        #
        # ptu.from_numpy converts NumPy arrays to torch float tensors and moves
        # them to ptu.device, which is CPU or GPU depending on init_gpu(...).
        obs = ptu.from_numpy(obs)
        actions = ptu.from_numpy(actions)
        advantages = ptu.from_numpy(advantages)

        # This is not another rollout in the environment. We are re-running the
        # policy network on the old collected observations to get pi_theta(.|s_t),
        # then evaluating the log-probability of the actions that were actually
        # sampled during rollout.
        action_distribution = self.forward(obs)
        if self.discrete:
            # CartPole example: actions [1.0, 0.0, 1.0] -> [1, 0, 1].
            actions = actions.long()

        # Example probs for actions [1, 0, 1]:
        # obs[0]: [left=0.4, right=0.6] -> log_probs[0] = log(0.6)
        # obs[1]: [left=0.7, right=0.3] -> log_probs[1] = log(0.7)
        # obs[2]: [left=0.2, right=0.8] -> log_probs[2] = log(0.8)
        log_probs = action_distribution.log_prob(actions)

        # Example:
        # log_probs = [log(0.6), log(0.7), log(0.8)]
        # advantages = [2.0, -1.0, 3.0]
        # Positive advantage -> make that sampled action more likely.
        # Negative advantage -> make that sampled action less likely.
        # Negative sign because PyTorch minimizes loss, but PG maximizes reward.
        loss = -(log_probs * advantages).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "Actor Loss": loss.item(),
        }
