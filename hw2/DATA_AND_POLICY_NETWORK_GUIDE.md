# HW2 Data and Policy Network Guide

This note explains how data flows through the HW2 policy-gradient code before the return, advantage, and policy-gradient update math happens.

It focuses on:

- trajectory sampling and shapes
- how rollout data is batched
- what the policy network outputs
- discrete vs continuous action distributions
- why the policy samples actions instead of always taking the largest score

The policy-gradient update details are intentionally left for a separate agent update workflow note.

---

## 1. Training Data Flow

### 1.1 Main Loop

The main training loop in `src/scripts/run.py` repeatedly collects a batch of environment interaction data, reorganizes it, and sends it into the agent update:

```python
trajs, envsteps_this_batch = utils.sample_trajectories(
    env, agent.actor, args.batch_size, max_ep_len
)

trajs_dict = {k: [traj[k] for traj in trajs] for k in trajs[0]}

train_info = agent.update(
    trajs_dict["observation"],
    trajs_dict["action"],
    trajs_dict["reward"],
    trajs_dict["terminal"],
)
```

At a high level:

1. `sample_trajectories(...)` collects rollouts using the current policy.
2. Each rollout is stored as one trajectory dictionary.
3. `trajs_dict` reorganizes the list of trajectories by field name.
4. `agent.update(...)` trains on the collected batch.

Important terms:

- A **trajectory** is one episode or rollout segment.
- A **batch** is the collection of many timesteps across one or more trajectories.
- `args.batch_size` means minimum number of environment steps to collect, not number of trajectories.

### 1.2 `sample_trajectories(...)`

The call:

```python
trajs, envsteps_this_batch = utils.sample_trajectories(
    env, agent.actor, args.batch_size, max_ep_len
)
```

keeps sampling full trajectories until it has at least `args.batch_size` timesteps:

```python
timesteps_this_batch = 0
trajs = []

while timesteps_this_batch < min_timesteps_per_batch:
    traj = sample_trajectory(env, policy, max_length, render)
    trajs.append(traj)
    timesteps_this_batch += get_traj_length(traj)
```

Example:

```text
args.batch_size = 1000

T_0 = 200
T_1 = 180
T_2 = 170
T_3 = 200
T_4 = 160
T_5 = 110
```

Then:

```text
envsteps_this_batch = 1020
```

The collected count can be slightly larger than `args.batch_size` because the code finishes the last trajectory before stopping.

### 1.3 One Environment Step

Inside `sample_trajectory(...)`, the main interaction is:

```python
ac = policy.get_action(ob)
step_result = env.step(ac)
```

This means:

1. The environment provides the current observation `ob`.
2. The policy samples one action `ac` from that observation.
3. `env.step(ac)` applies that action for one timestep.
4. The environment returns the next observation, reward, and done information.

The policy does not choose the whole trajectory at once. It chooses one action at a time:

```text
ob_t -> policy -> ac_t -> env.step(ac_t) -> ob_{t+1}, reward_t, done_t
```

Then the loop continues from `ob_{t+1}`.

### 1.4 Gym API Details

Older Gym returns only the observation from `reset()`:

```python
ob = env.reset()
```

Newer Gym/Gymnasium returns:

```python
ob, info = env.reset()
```

That is why the code handles tuples:

```python
ob = env.reset()
if isinstance(ob, tuple):
    ob = ob[0]
```

Similarly, older Gym returns four values from `step(...)`:

```python
next_ob, rew, done, info = env.step(ac)
```

Newer Gym/Gymnasium returns five:

```python
next_ob, rew, terminated, truncated, info = env.step(ac)
done = terminated or truncated
```

The code supports both:

```python
step_result = env.step(ac)
if len(step_result) == 5:
    next_ob, rew, terminated, truncated, info = step_result
    done = terminated or truncated
else:
    next_ob, rew, done, info = step_result
```

Meanings:

| Variable | Meaning |
|---|---|
| `next_ob` | next observation after applying the action |
| `rew` | reward for this one timestep |
| `done` | whether the episode/rollout ended |
| `terminated` | natural environment termination |
| `truncated` | time-limit or wrapper cutoff |
| `info` | extra environment metadata |

---

## 2. Trajectory Data Structures

### 2.1 One Transition

After one environment step, the code records:

```python
obs.append(ob)
acs.append(ac)
rewards.append(rew)
next_obs.append(next_ob)
terminals.append(rollout_done)
```

This stores one transition:

```text
(ob_t, ac_t, reward_t, next_ob_t, terminal_t)
```

For `CartPole-v0`:

| Variable | Shape | Meaning |
|---|---:|---|
| `ob` | `(4,)` | cart position, cart velocity, pole angle, pole angular velocity |
| `ac` | scalar | action ID, either `0` or `1` |
| `rew` | scalar | reward for this step |
| `next_ob` | `(4,)` | next CartPole state |
| `rollout_done` | scalar bool | whether this is the final recorded step |

In general:

```text
ob.shape       = env.observation_space.shape
next_ob.shape  = env.observation_space.shape
```

Action shape depends on the action space:

```text
discrete action:   scalar action ID
continuous action: shape (ac_dim,)
```

### 2.2 `rollout_done`

The code computes:

```python
rollout_done = done or steps >= max_length
```

The rollout stops if either:

1. the environment says the episode is done
2. the rollout reaches the maximum allowed episode length

Then:

```python
if rollout_done:
    break
```

`rollout_done` is stored in the trajectory as the `terminal` field. It marks the end of that trajectory.

### 2.3 One Trajectory Dictionary

At the end of `sample_trajectory(...)`, the lists are converted into arrays:

```python
return {
    "observation": np.array(obs, dtype=np.float32),
    "image_obs": np.array(image_obs, dtype=np.uint8),
    "reward": np.array(rewards, dtype=np.float32),
    "action": np.array(acs, dtype=np.float32),
    "next_observation": np.array(next_obs, dtype=np.float32),
    "terminal": np.array(terminals, dtype=np.float32),
}
```

If a trajectory has length `T`, then for CartPole:

```text
observation.shape       = (T, 4)
action.shape            = (T,)
reward.shape            = (T,)
next_observation.shape  = (T, 4)
terminal.shape          = (T,)
```

For a continuous action environment:

```text
action.shape = (T, ac_dim)
```

The full returned object is:

```python
trajs = [
    traj_0,
    traj_1,
    traj_2,
    ...
]
```

where each `traj_i` is one trajectory dictionary.

### 2.4 `image_obs`

`image_obs` is used only for video logging:

```python
eval_video_trajs = utils.sample_n_trajectories(
    env, agent.actor, MAX_NVIDEO, max_ep_len, render=True
)
```

During normal training, `render=False`, so `image_obs` remains empty.

If `render=True`, the code renders and resizes each frame:

```python
image_obs.append(
    cv2.resize(img, dsize=(250, 250), interpolation=cv2.INTER_CUBIC)
)
```

One image frame has shape:

```text
(250, 250, 3)
```

If a rendered trajectory has length `T`:

```text
image_obs.shape = (T, 250, 250, 3)
```

If `render=False`:

```text
image_obs.shape = (0,)
```

`image_obs` is not used for policy training in this homework.

---

## 3. Batching for `agent.update(...)`

### 3.1 Why `trajs[0]` Is Used

This line reorganizes the trajectory list:

```python
trajs_dict = {k: [traj[k] for traj in trajs] for k in trajs[0]}
```

`trajs[0]` is used only to get the dictionary keys:

```text
"observation"
"image_obs"
"reward"
"action"
"next_observation"
"terminal"
```

It does not mean the code trains only on the first trajectory.

The inner expression:

```python
[traj[k] for traj in trajs]
```

collects the same field from every trajectory.

So this converts:

```python
[
    {"observation": obs_0, "action": acs_0, "reward": rew_0},
    {"observation": obs_1, "action": acs_1, "reward": rew_1},
]
```

into:

```python
{
    "observation": [obs_0, obs_1],
    "action": [acs_0, acs_1],
    "reward": [rew_0, rew_1],
}
```

### 3.2 Shapes Passed Into `agent.update(...)`

The update receives lists of per-trajectory arrays:

```python
agent.update(
    trajs_dict["observation"],
    trajs_dict["action"],
    trajs_dict["reward"],
    trajs_dict["terminal"],
)
```

If trajectory `i` has length `T_i`, then:

```text
observation[i].shape = (T_i, ob_dim)
action[i].shape      = (T_i,) for discrete
action[i].shape      = (T_i, ac_dim) for continuous
reward[i].shape      = (T_i,)
terminal[i].shape    = (T_i,)
```

`T_i` is the length of one trajectory. It is not the full update batch size.

The update batch size is:

```text
batch_size = sum_i T_i
```

### 3.3 Flattening the Batch

Inside `PGAgent.update(...)`, the lists are flattened:

```python
obs = np.concatenate(obs)
actions = np.concatenate(actions)
rewards = np.concatenate(rewards)
terminals = np.concatenate(terminals)
```

After concatenation:

```text
obs.shape       = (batch_size, ob_dim)
actions.shape   = (batch_size,) for discrete
actions.shape   = (batch_size, ac_dim) for continuous
rewards.shape   = (batch_size,)
terminals.shape = (batch_size,)
```

For CartPole:

```text
ob_dim = 4
ac_dim = 2
```

but because CartPole is discrete:

```text
obs.shape     = (batch_size, 4)
actions.shape = (batch_size,)
```

Each action is an ID, not a two-dimensional vector.

---

## 4. Policy Network

### 4.1 What `forward(...)` Returns

The policy network lives in `src/networks/policies.py`.

It maps observations to an action distribution:

```text
observation -> neural network -> distribution over actions
```

It does not directly return the final action from `forward(...)`.

Instead:

```python
action_distribution = self.forward(obs)
action = action_distribution.sample()
```

This is important because vanilla policy gradient uses a stochastic policy.

The policy needs to:

1. sample actions during rollout
2. compute `log_prob(action)` during training

### 4.2 Discrete Action Policy

Discrete action spaces contain a fixed set of action IDs.

CartPole is discrete:

```text
action 0 = push left
action 1 = push right
```

The policy builds:

```python
self.logits_net = ptu.build_mlp(
    input_size=ob_dim,
    output_size=ac_dim,
    n_layers=n_layers,
    size=layer_size,
)
```

For CartPole:

```text
ob_dim = 4
ac_dim = 2
```

So:

```text
input shape  = (batch_size, 4)
output shape = (batch_size, 2)
```

The output is called `logits`.

### 4.3 Logits

Logits are raw relative scores before softmax.

Example:

```python
logits = [1.2, 0.3]
```

This means action `0` has a higher score than action `1`.

The logits are converted into probabilities by softmax:

```text
p_i = exp(logit_i) / sum_j exp(logit_j)
```

For example:

```text
softmax([1.2, 0.3]) ≈ [0.71, 0.29]
```

Only relative differences matter:

```text
softmax([1.2, 0.3]) == softmax([101.2, 100.3])
```

because both pairs have the same difference between action scores.

The word "logit" comes from logistic regression, where the binary logit is:

```text
log(p / (1 - p))
```

For multiple actions, the network outputs a vector of relative action scores. In the binary case, the difference between two logits acts like the binary log-odds.

### 4.4 `D.Categorical`

The code uses:

```python
return D.Categorical(logits=logits)
```

`D.Categorical` is PyTorch's distribution over a finite set of choices:

```text
0, 1, 2, ..., ac_dim - 1
```

It can handle two actions, but it is not only a binary distribution.

For a batch of observations:

```text
logits.shape = (batch_size, ac_dim)
```

Sampling returns one action ID per row:

```text
sample().shape = (batch_size,)
```

Example:

```python
logits = [
    [1.2, 0.3],
    [-0.5, 2.0],
    [0.1, 0.1],
]

actions = dist.sample()
# possible result: [0, 1, 0]
```

There is no `ac_dim` dimension in the sampled discrete action because each sampled action is one integer ID.

### 4.5 Continuous Action Policy

Continuous action spaces use real-valued actions.

A continuous action can be one number or a vector. Gym usually represents it as a `Box` with shape:

```text
(ac_dim,)
```

If:

```text
ac_dim = 1
```

then one action has shape `(1,)`, for example:

```python
action = np.array([0.37])
```

If:

```text
ac_dim = 3
```

then one action has shape `(3,)`, for example:

```python
action = np.array([0.2, -0.5, 1.1])
```

The continuous policy builds:

```python
self.mean_net = ptu.build_mlp(
    input_size=ob_dim,
    output_size=ac_dim,
    n_layers=n_layers,
    size=layer_size,
)

self.logstd = nn.Parameter(
    torch.zeros(ac_dim, dtype=torch.float32, device=ptu.device)
)
```

The network predicts the Gaussian mean:

```text
mean.shape = (batch_size, ac_dim)
```

`logstd` is a learned parameter:

```text
logstd.shape = (ac_dim,)
```

Then:

```python
std = torch.exp(self.logstd)
```

ensures the standard deviation is positive.

### 4.6 Continuous Gaussian Distribution

The code returns:

```python
return D.Independent(D.Normal(mean, std), 1)
```

`D.Normal(mean, std)` creates one Normal distribution per action dimension.

Without `D.Independent`, log-probabilities would have one value per action dimension:

```text
D.Normal(mean, std).log_prob(action).shape = (batch_size, ac_dim)
```

But for policy gradient, we want one log-probability per full action vector:

```text
log_prob(action).shape = (batch_size,)
```

`D.Independent(..., 1)` treats the last dimension, `ac_dim`, as part of one full action event. It sums the per-dimension log-probabilities into one log-probability per action vector.

For continuous actions:

```text
sample().shape         = (batch_size, ac_dim)
log_prob(action).shape = (batch_size,)
```

For discrete actions:

```text
sample().shape         = (batch_size,)
log_prob(action).shape = (batch_size,)
```

### 4.7 `get_action(...)` Shapes

`get_action(...)` takes one observation from the environment:

```python
def get_action(self, obs: np.ndarray) -> np.ndarray:
```

For CartPole:

```text
obs.shape = (4,)
```

The policy network expects a batch dimension, so the code does:

```python
if obs.ndim == 1:
    obs = obs[None]
```

Now:

```text
obs.shape = (1, ob_dim)
```

Then:

```python
action_distribution = self.forward(ptu.from_numpy(obs))
action = action_distribution.sample()
```

Before removing the batch dimension:

```text
discrete action sample shape   = (1,)
continuous action sample shape = (1, ac_dim)
```

Finally:

```python
return ptu.to_numpy(action)[0]
```

After removing the batch dimension:

```text
discrete action returned   = scalar action ID
continuous action returned = shape (ac_dim,)
```

Examples:

```python
# CartPole discrete action
ac = 1

# continuous action with ac_dim = 3
ac = np.array([0.2, -0.5, 1.1])
```

### 4.8 Why Sample Instead of Taking `argmax`

The policy could choose:

```python
action = torch.argmax(logits, dim=-1)
```

but that would be deterministic.

During policy-gradient training, we usually want a stochastic policy:

```python
action = dist.sample()
```

Example:

```text
logits = [1.2, 0.3]
probs  = [0.71, 0.29]
```

Sampling means:

```text
71% chance: action 0
29% chance: action 1
```

This exploration matters because the agent needs to try actions that are not currently the most likely.

Policy gradient also needs:

```python
log_prob = dist.log_prob(action)
```

The update uses the log-probability of the action that was actually sampled.

### 4.9 Identity Output Layer

The helper `ptu.build_mlp(...)` has:

```python
output_activation="identity"
```

An identity layer means:

```text
Identity(x) = x
```

In PyTorch:

```python
nn.Identity()
```

returns the input unchanged.

This is useful because the final policy outputs should be raw values:

- discrete policy: raw logits, not already-softmaxed probabilities
- continuous policy: raw Gaussian means, not clipped through `relu`, `sigmoid`, or `tanh`

So the last linear layer output is left unchanged.

---

## 5. Summary

The HW2 data flow before the update is:

```text
environment observation
    -> policy.get_action(...)
    -> sampled action
    -> env.step(action)
    -> transition data
    -> trajectory dictionary
    -> list of trajectories
    -> trajs_dict
    -> agent.update(...)
```

The main shape rules are:

```text
One trajectory:
observation       (T_i, ob_dim)
action            (T_i,) for discrete, (T_i, ac_dim) for continuous
reward            (T_i,)
terminal          (T_i,)

Flattened update batch:
observation       (batch_size, ob_dim)
action            (batch_size,) for discrete, (batch_size, ac_dim) for continuous
reward            (batch_size,)
terminal          (batch_size,)
```

The policy network returns distributions:

```text
discrete      -> Categorical(logits)
continuous    -> Gaussian action-vector distribution
```

Sampling from those distributions gives the action sent to `env.step(...)`. Computing `log_prob(action)` from those same distributions is what later allows the policy-gradient update to increase or decrease the probability of sampled actions.
