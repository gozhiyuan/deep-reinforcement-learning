# SAC Implementation Guide

A practical guide to how the HW3 Soft Actor-Critic implementation works, how data moves through replay, and how the actor and critic are updated.

---

## Overview

This project implements **Soft Actor-Critic (SAC)** for **continuous action spaces**.

SAC is an off-policy actor-critic method:

```text
actor:
  learns a stochastic policy pi(a | s)
  outputs a distribution over continuous actions

critic:
  learns Q(s, a)
  scores state-action pairs

replay buffer:
  stores old transitions
  lets the agent train from random minibatches
```

Relevant files:

- `src/scripts/run_sac.py`: main SAC training loop
- `src/agents/sac_agent.py`: actor, critic, target critic, and update logic
- `src/networks/policies.py`: `MLPPolicy`, used as the SAC actor
- `src/networks/critics.py`: `StateActionCritic`, used as the SAC critic
- `src/infrastructure/distributions.py`: Gaussian and tanh-squashed Gaussian helpers
- `src/infrastructure/replay_buffer.py`: replay buffer
- `src/configs/sac_config.py`: SAC configs and environment wrappers
- `experiments/sac/*.yaml`: experiment hyperparameters

Run command:

```bash
uv run src/scripts/run_sac.py -cfg experiments/sac/<CONFIG>.yaml
```

---

## 1. SAC vs DQN

DQN is value-only:

```text
critic(obs) -> Q-value for each discrete action
action = argmax_a Q(obs, a)
```

SAC has an explicit actor and critic:

```text
actor(obs) -> action distribution
sample action from distribution
critic(obs, action) -> Q-value for that state-action pair
```

Why SAC needs an actor:

```text
DQN:
  max over discrete actions is easy
  example: max over [left, right]

SAC:
  actions are continuous
  example: torque vector [0.4, -0.2, 0.7]
  max over all real-valued actions is hard
```

The actor solves this by directly proposing continuous actions.

---

## 2. Environment Data Shapes

SAC configs in this homework use MuJoCo vector observations, not images. There is no CNN encoder in this SAC path.

Examples:

```text
InvertedPendulum-v4:
  observation.shape = (4,)
  action.shape      = (1,)

Hopper-v4:
  observation.shape = (11,)
  action.shape      = (3,)

HalfCheetah-v4:
  observation.shape = (17,)
  action.shape      = (6,)
```

With `batch_size = 128`, replay samples look like:

```text
InvertedPendulum-v4:
  observations:      (128, 4)
  actions:           (128, 1)
  rewards:           (128,)
  next_observations: (128, 4)
  dones:             (128,)

Hopper-v4:
  observations:      (128, 11)
  actions:           (128, 3)
  rewards:           (128,)
  next_observations: (128, 11)
  dones:             (128,)
```

`reward` is a scalar per transition. `done` is a scalar bool per transition.

---

## 3. Training Loop

The outer loop in `run_sac.py` is similar to DQN:

```text
for each environment step:
  choose action
  env.step(action)
  store transition in replay buffer
  after training_starts, sample replay batch
  update SAC agent
```

Code shape:

```python
if step < config["random_steps"]:
    action = env.action_space.sample()
else:
    action = agent.get_action(observation)

next_observation, reward, done, info = env.step(action)

replay_buffer.insert(
    observation=observation,
    action=action,
    reward=reward,
    next_observation=next_observation,
    done=done and not info.get("TimeLimit.truncated", False),
)

if step >= config["training_starts"]:
    batch = replay_buffer.sample(config["batch_size"])
    batch = ptu.from_numpy(batch)
    update_info = agent.update(...)
```

Important point:

```text
The env loop collects one new transition.
The update uses a random batch from replay, not just the newest transition.
```

---

## 4. Actor Network

The SAC actor is `MLPPolicy` from `src/networks/policies.py`.

For continuous SAC:

```text
obs -> MLP -> mean and std -> action distribution
```

If `state_dependent_std=True`, the network outputs both mean and raw std:

```text
network output shape = (batch_size, 2 * action_dim)

split into:
  mean:    (batch_size, action_dim)
  raw_std: (batch_size, action_dim)
```

Example Hopper:

```text
obs.shape = (128, 11)
action_dim = 3

self.net(obs).shape = (128, 6)
mean.shape          = (128, 3)
raw_std.shape       = (128, 3)
```

The raw std is converted to a positive std:

```python
std = torch.nn.functional.softplus(raw_std) + 1e-2
```

Why:

```text
MLP outputs can be negative.
Standard deviation must be positive.
softplus(x) maps any real number to a positive value.
```

---

## 5. Action Distributions

The actor returns a PyTorch distribution, not a raw action tensor.

In SAC with `use_tanh=True`:

```python
action_distribution = make_tanh_transformed(mean, std)
```

This creates:

```text
raw_action ~ Normal(mean, std)
action = tanh(raw_action)
```

The final action is in:

```text
(-1, 1)
```

That matches the environment wrappers in `sac_config.py`:

```python
RescaleAction(env, -1, 1)
ClipAction(...)
```

### 5.1 Why `Independent`

For a continuous action vector:

```text
action = [a1, a2, a3]
```

SAC needs one log probability for the full vector:

```text
log pi([a1, a2, a3] | obs)
```

Without `torch.distributions.Independent`:

```text
log_prob.shape = (batch_size, action_dim)
```

With `Independent(..., reinterpreted_batch_ndims=1)`:

```text
log_prob.shape = (batch_size,)
```

It sums log probabilities over action dimensions:

```text
log pi([a1, a2, a3] | s)
= log pi(a1 | s) + log pi(a2 | s) + log pi(a3 | s)
```

Sampling shape does not change:

```text
sample.shape = (batch_size, action_dim)
```

### 5.2 Diagonal Gaussian Policy

A diagonal Gaussian policy assumes action dimensions are independent given the state:

```text
a1 ~ Normal(mean_1, std_1)
a2 ~ Normal(mean_2, std_2)
a3 ~ Normal(mean_3, std_3)
```

There is no learned covariance between action dimensions. This is simpler than a full multivariate Gaussian and is standard in many SAC implementations.

---

## 6. `sample()` vs `rsample()`

Both produce random actions.

`sample()`:

```text
random action
no gradient through the sampled action
good for env rollout and critic targets
```

`rsample()`:

```text
reparameterized random action
keeps gradient path through mean/std
needed for actor update
```

For a Gaussian:

```text
epsilon ~ Normal(0, 1)
action = mean + std * epsilon
```

With `rsample()`, PyTorch keeps the graph:

```text
loss -> action -> mean/std -> actor weights
```

Usage in this code:

```python
# environment rollout
with torch.no_grad():
    action = action_distribution.sample()

# actor update
action = action_distribution.rsample()
```

---

## 7. Critic Network

The SAC critic is `StateActionCritic`.

It maps:

```text
(obs, action) -> Q(obs, action)
```

Example InvertedPendulum:

```text
obs.shape    = (128, 4)
action.shape = (128, 1)

critic input after concat = (128, 5)
critic output             = (128,)
```

The SAC agent stacks critic outputs:

```python
return torch.stack([critic(obs, action) for critic in self.critics], dim=0)
```

So if `num_critic_networks=2`:

```text
self.critic(obs, action).shape = (2, 128)
```

---

## 8. Target Critics

SAC uses target critics, like DQN uses a target Q-network.

If there are multiple critics, there are multiple target critics:

```text
num_critic_networks = 1:
  critics:        [Q1]
  target_critics: [Q1_target]

num_critic_networks = 2:
  critics:        [Q1, Q2]
  target_critics: [Q1_target, Q2_target]
```

Target critics are delayed copies used for stable bootstrapped targets.

Soft target update:

```text
target <- (1 - tau) * target + tau * current
```

Hard target update:

```text
target <- current
```

---

## 9. Critic Update

The critic update is Bellman regression.

Replay batch:

```text
obs, action, reward, next_obs, done
```

Critic prediction:

```python
q_values = self.critic(obs, action)
```

Shape:

```text
q_values.shape = (num_critics, batch_size)
```

Target construction:

```python
with torch.no_grad():
    next_action_distribution = self.actor(next_obs)
    next_action = next_action_distribution.sample()
    next_qs = self.target_critic(next_obs, next_action)
```

Meaning:

```text
At next_obs, ask the current actor what action it would take.
Use target critic to score that next action.
```

If entropy backup is enabled:

```python
next_action_entropy = self.entropy(next_action_distribution)
next_qs = next_qs + temperature * next_action_entropy[None]
```

Final target:

```python
target_values = reward[None] + discount * (1 - done.float())[None] * next_qs
```

Loss:

```python
loss = MSE(q_values, target_values)
```

Optimizer step:

```python
critic_optimizer.zero_grad()
loss.backward()
critic_optimizer.step()
```

### 9.1 Why `torch.no_grad()` Here

During critic update, the target side should be fixed.

The actor and target critic are used to compute:

```text
bootstrap target
```

but they are not updated by critic loss.

Only current critic weights are updated.

---

## 10. Entropy and Temperature

SAC uses a maximum-entropy objective:

```text
maximize reward + temperature * entropy
```

Entropy:

```text
H(pi(. | s)) = E[-log pi(a | s)]
```

Approximate entropy in code:

```python
action = action_distribution.rsample()
entropy = -action_distribution.log_prob(action)
```

Shape:

```text
entropy.shape = (batch_size,)
```

Temperature controls the reward-vs-randomness tradeoff:

```text
temperature = 0:
  ignore entropy

larger temperature:
  stronger exploration/randomness bonus
```

`backup_entropy` controls whether entropy is included in the critic target:

```text
backup_entropy=True:
  target uses Q_target + temperature * entropy

backup_entropy=False:
  target uses Q_target only
```

---

## 11. Automatic Temperature Tuning

Fixed temperature means the config chooses one entropy weight:

```text
temperature = 0.1
```

Automatic tuning learns that value during training.

The goal is:

```text
keep policy entropy near a target level
```

For continuous actions, this code uses:

```python
self.target_entropy = -float(action_dim)
```

Examples:

```text
InvertedPendulum action_dim=1:
  target_entropy = -1

Hopper action_dim=3:
  target_entropy = -3

HalfCheetah action_dim=6:
  target_entropy = -6
```

### 11.1 Why Learn `log_alpha`

Temperature must stay positive:

```text
alpha > 0
```

So the code learns `log_alpha`:

```python
self.log_alpha = nn.Parameter(torch.tensor(np.log(temperature)))
```

and converts it back with:

```python
alpha = self.log_alpha.exp()
```

This guarantees:

```text
alpha is always positive
```

### 11.2 Where Alpha Is Used

When auto-tuning is enabled:

```python
self.get_temperature()
```

returns:

```python
exp(log_alpha)
```

That learned alpha is used in the same places fixed temperature was used:

```text
critic target:
  Q_target + alpha * entropy

actor loss:
  -Q - alpha * entropy
```

### 11.3 Alpha Update Inputs and Shapes

Actor update computes:

```python
log_prob = action_distribution.log_prob(action)
```

Shape:

```text
log_prob.shape = (batch_size,)
```

This is:

```text
log pi(action | obs)
```

Entropy estimate:

```text
entropy ~= -log_prob
```

Then `update_alpha(log_prob)` updates only `log_alpha`.

### 11.4 Alpha Loss

The implemented loss is:

```python
alpha = self.log_alpha.exp()
alpha_loss = -(alpha * (log_prob.detach() + self.target_entropy)).mean()
```

The `detach()` matters:

```text
alpha update should change alpha only
not actor weights
```

Optimizer:

```python
self.alpha_optimizer.zero_grad()
alpha_loss.backward()
self.alpha_optimizer.step()
```

### 11.5 Sign Intuition

Remember:

```text
entropy ~= -log_prob
```

If entropy is too low:

```text
policy is too deterministic
alpha should increase
entropy bonus gets stronger
actor is pushed to be more random
```

If entropy is too high:

```text
policy is too random
alpha should decrease
entropy bonus gets weaker
actor can focus more on Q/reward
```

So alpha is a learned knob:

```text
high alpha:
  stronger exploration pressure

low alpha:
  stronger reward-seeking pressure
```

### 11.6 Full Auto-Tune Workflow

Each SAC update does:

```text
1. update critic using current alpha in the entropy backup
2. update actor using current alpha in actor loss
3. update alpha using actor log_prob and target_entropy
4. update target critic
```

In code:

```python
critic_infos = update_critic(...)
actor_info = update_actor(...)
alpha_info = update_alpha(actor_info["log_prob"])
soft_update_target_critic(...)
```

Logged values:

```text
temperature:
  current alpha, fixed or learned

alpha:
  learned alpha when auto-tuning is enabled

alpha_loss:
  dual loss used to update alpha
```

---

## 12. Actor Update

The actor update uses replay observations, not replay actions.

Input:

```text
obs: (batch_size, obs_dim)
```

Step 1: get actor distribution:

```python
action_distribution = self.actor(obs)
```

Step 2: sample fresh actor actions:

```python
action = action_distribution.rsample()
```

Shape:

```text
action.shape = (batch_size, action_dim)
```

Step 3: critic scores actor actions:

```python
q_values = self.critic(obs, action)
```

Shape:

```text
q_values.shape = (num_critics, batch_size)
```

Step 4: combine critics:

```python
q_actor = self.q_backup_strategy(q_values)
```

Step 5: actor loss:

```python
loss = -q_actor.mean()
```

Then entropy is added in `update_actor()`:

```python
loss = loss - temperature * entropy
```

Final actor loss:

```text
actor_loss = -Q(obs, actor_action) - temperature * entropy
```

Minimizing this is equivalent to maximizing:

```text
Q(obs, actor_action) + temperature * entropy
```

### 12.1 Why Actor Does Not Use Replay Actions

Replay actions answer:

```text
what action was taken in the past?
```

Actor update asks:

```text
what action should the current actor take now?
```

So actor samples new actions from itself and uses the critic as feedback.

---

## 13. Multiple Critics and `q_backup_strategy`

`q_backup_strategy()` combines critic estimates.

Input example:

```text
num_critics = 2
batch_size = 3

next_qs.shape = (2, 3)

next_qs =
[
  [10.0, 5.0, 8.0],   # Q1
  [ 9.0, 7.0, 6.0],   # Q2
]
```

Mean strategy:

```text
mean = [9.5, 6.0, 7.0]
```

Min strategy:

```text
min = [9.0, 5.0, 6.0]
```

`min` is clipped double-Q. It is conservative and reduces overestimated Q-values.

After combining, the result is expanded back:

```text
(batch_size,) -> (num_critics, batch_size)
```

Why:

```text
each critic is trained against the same combined target
```

---

## 14. One Full SAC Update

Given replay batch:

```text
obs, action, reward, next_obs, done
```

Critic update:

```text
1. q_values = critic(obs, replay_action)
2. next_action = actor(next_obs).sample()
3. next_qs = target_critic(next_obs, next_action)
4. optionally add temperature * entropy
5. target = reward + discount * (1 - done) * next_qs
6. train critic with MSE(q_values, target)
```

Actor update:

```text
1. action = actor(obs).rsample()
2. q_values = critic(obs, action)
3. entropy = -log pi(action | obs)
4. actor_loss = -Q - temperature * entropy
5. train actor with actor_loss
```

Target critic update:

```text
soft:
  target <- (1 - tau) * target + tau * critic

hard:
  target <- critic every target_update_period steps
```

---

## 15. Mental Model

Critic:

```text
learns from replay transitions
answers: how good is this state-action pair?
```

Actor:

```text
learns from critic feedback
answers: what action should I sample in this state?
```

Entropy:

```text
keeps the actor stochastic
helps exploration in continuous action spaces
```

Target critics:

```text
slow-moving copies
make bootstrapped critic targets more stable
```

Replay buffer:

```text
stores past experience
lets SAC train off-policy from random minibatches
```
