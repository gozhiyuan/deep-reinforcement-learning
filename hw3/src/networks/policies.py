from typing import Optional

from torch import nn

import torch
from torch import distributions

from infrastructure import pytorch_util as ptu
from infrastructure.distributions import make_tanh_transformed, make_multi_normal


class MLPPolicy(nn.Module):
    """
    Base MLP policy, which can take an observation and output a distribution over actions.

    This is an explicit policy/actor network. SAC uses it for the actor:
        observation -> action distribution -> sampled continuous action

    DQN also uses ptu.build_mlp, but not through this policy class. DQN's MLP
    lives in DQNCritic and outputs Q-values for discrete actions, not an action
    distribution.

    This class implements `forward()` which takes a (batched) observation and returns a distribution over actions.
    """

    def __init__(
        self,
        ac_dim: int,
        ob_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        use_tanh: bool = False,
        state_dependent_std: bool = False,
        fixed_std: Optional[float] = None,
    ):
        super().__init__()

        self.use_tanh = use_tanh
        self.discrete = discrete
        self.state_dependent_std = state_dependent_std
        self.fixed_std = fixed_std

        if discrete:
            # Discrete policy: output one logit per action.
            # Example with CartPole: obs_dim=4, ac_dim=2, so this network maps
            # obs -> [logit_left, logit_right]. forward() wraps these logits in
            # a Categorical distribution.
            self.logits_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            )
        else:
            if self.state_dependent_std:
                assert fixed_std is None
                # Continuous stochastic policy with state-dependent mean and
                # std. The network outputs both vectors at once:
                #   obs -> [mean_1..mean_ac_dim, raw_std_1..raw_std_ac_dim]
                # For Hopper, ac_dim=3, so output_size=6.
                self.net = ptu.build_mlp(
                    input_size=ob_dim,
                    output_size=2*ac_dim,
                    n_layers=n_layers,
                    size=layer_size,
                )
            else:
                # Continuous stochastic policy with state-dependent mean only.
                # The action std is either a fixed scalar or a learned vector
                # shared across all observations.
                self.net = ptu.build_mlp(
                    input_size=ob_dim,
                    output_size=ac_dim,
                    n_layers=n_layers,
                    size=layer_size,
                )

                if self.fixed_std:
                    # Fixed exploration scale for every action dimension.
                    self.std = 0.1
                else:
                    # Learned global raw std parameter. softplus() in forward()
                    # converts it to a positive standard deviation.
                    self.std = nn.Parameter(
                        torch.full((ac_dim,), 0.0, dtype=torch.float32, device=ptu.device)
                    )

    def forward(self, obs: torch.FloatTensor) -> distributions.Distribution:
        """
        This function defines the forward pass of the network.  You can return anything you want, but you should be
        able to differentiate through it. For example, you can return a torch.FloatTensor. You can also return more
        flexible objects, such as a `torch.distributions.Distribution` object. It's up to you!
        """
        if self.discrete:
            # logits shape: (batch_size, ac_dim). Categorical.sample() returns
            # integer action IDs with shape (batch_size,).
            logits = self.logits_net(obs)
            action_distribution = distributions.Categorical(logits=logits)
        else:
            if self.state_dependent_std:
                # self.net(obs) shape: (batch_size, 2 * ac_dim).
                # Split into mean and raw std vectors, each (batch_size, ac_dim).
                # The network's second half is not a valid std yet because MLP
                # outputs can be negative. softplus(raw_std) smoothly maps any
                # real number to a positive scale, and 1e-2 keeps the Gaussian
                # from collapsing to zero variance.
                mean, std = torch.chunk(self.net(obs), 2, dim=-1)
                std = torch.nn.functional.softplus(std) + 1e-2
            else:
                # mean shape: (batch_size, ac_dim). std is either a scalar or a
                # learned vector with shape (ac_dim,), broadcast over the batch.
                mean = self.net(obs)
                if self.fixed_std:
                    std = self.std
                else:
                    # Same positive-scale conversion for the learned global
                    # std parameter. Here std does not depend on obs; every
                    # state shares one learned std per action dimension.
                    std = torch.nn.functional.softplus(self.std) + 1e-2

            if self.use_tanh:
                # SAC actions are normalized to [-1, 1] by sac_config's action
                # wrappers, so tanh-squash the Gaussian samples into that range.
                action_distribution = make_tanh_transformed(mean, std)
            else:
                return make_multi_normal(mean, std)

        return action_distribution

    def get_action(self, obs: torch.FloatTensor) -> torch.Tensor:
        """
        Sample an action from the policy.

        Args:
            obs: (ob_dim,) or (batch_size, ob_dim) observation
        Returns:
            action: (ac_dim,) or (batch_size, ac_dim) action
        """
        dist = self.forward(obs)
        return dist.sample()
