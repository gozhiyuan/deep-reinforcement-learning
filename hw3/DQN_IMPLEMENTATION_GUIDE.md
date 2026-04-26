# DQN Implementation Guide

A practical guide to how the HW3 DQN implementation works, how data moves through the replay buffer, and how the Q-network is updated.

---

## Overview

This project implements **DQN**, a value-based reinforcement learning algorithm for **discrete action spaces**.

### Implementation Summary

This implementation combines ideas from online Q-learning and fitted/iterative Q-learning:

```text
online Q-learning side:
  - the agent continuously interacts with the environment
  - new transitions are collected step by step
  - the behavior policy changes over training through epsilon-greedy actions

fitted Q-iteration side:
  - updates look like supervised regression with MSE loss
  - targets are Bellman-style TD targets
  - training uses random minibatches from a replay buffer instead of only the newest transition
```

A concise label is:

```text
serial online DQN with replay-buffered fitted TD updates
```

This implementation is **synchronous/serial**, not parallel or asynchronous:

```text
one environment
one replay buffer
one current critic
one target critic
one learner loop
```

There are no rollout workers, no parameter server, and no asynchronous gradient pushes. The target critic is a delayed copy used for stable Bellman targets; it is not an async worker model.

Other key choices:

```text
action space: discrete only
policy: implicit argmax over critic Q-values, plus epsilon exploration
target: one-step TD target
target network update: hard copy every target_update_period steps
Double DQN: optional, separates next-action selection and evaluation
```

The implementation is best described as:

```text
online data collection
+ experience replay
+ one-step TD targets
+ target network
+ optional Double DQN
```

It is not pure fitted Q-iteration, and it is not pure online tabular Q-learning. It collects experience online from the environment, stores transitions in a replay buffer, then trains the Q-network using random minibatches sampled from that buffer.

Relevant files:

- `src/scripts/run_dqn.py`: main training loop
- `src/agents/dqn_agent.py`: DQN action selection and update logic
- `src/infrastructure/replay_buffer.py`: normal and memory-efficient replay buffers
- `src/configs/dqn_config.py`: network/config/schedule definitions
- `src/configs/schedule.py`: epsilon and learning-rate schedules

---

## 1. Big Picture

### 1.1 What Networks Does DQN Use?

DQN does **not** use a separate policy network.

Instead, it trains a Q-network, called the `critic` in this code:

```python
self.critic = make_critic(observation_shape, num_actions)
self.target_critic = make_critic(observation_shape, num_actions)
```

The critic takes an observation and outputs one Q-value per discrete action:

```python
qa_values = self.critic(obs)
```

Example with 4 actions:

```text
critic(obs) = [Q(obs, action_0), Q(obs, action_1), Q(obs, action_2), Q(obs, action_3)]
```

The policy is implicit:

```text
choose action = argmax_a Q(obs, a)
```

With epsilon-greedy exploration, the agent sometimes ignores the argmax and takes a random action.

#### Discrete Actions Only

This implementation requires a discrete action space:

```python
discrete = isinstance(env.action_space, gym.spaces.Discrete)
assert discrete, "DQN only supports discrete action spaces"
```

DQN cannot directly handle continuous actions because it outputs one Q-value per action. For continuous actions, there are infinitely many possible actions, so algorithms like SAC, TD3, or DDPG are usually used instead.

---

### 1.2 Old Gym API Used Here

HW3 pins:

```toml
gym==0.25.2
```

The DQN training loop is written for the old Gym API:

```python
observation = env.reset()
next_observation, reward, done, info = env.step(action)
```

New Gymnasium-style APIs return:

```python
observation, info = env.reset()
next_observation, reward, terminated, truncated, info = env.step(action)
```

This project intentionally expects the old format:

```python
assert not isinstance(observation, tuple)
```

The old API combines natural termination and time-limit cutoff into `done`, so the code recovers time-limit truncation from `info`:

```python
truncated = info.get("TimeLimit.truncated", False)
```

Then it stores:

```python
done=done and not truncated
```

This means:

```text
true terminal state      -> done=True
time-limit truncation    -> done=False for learning target
```

For Q-learning, a time-limit cutoff should usually not erase the bootstrap future value, because it is an artificial wrapper boundary rather than a true terminal state.

---

### 1.3 Epsilon-Greedy Exploration

Action selection happens in `DQNAgent.get_action`:

```python
if np.random.random() < epsilon:
    return int(np.random.randint(self.num_actions))

qa_values = self.critic(observation)
action = torch.argmax(qa_values, dim=1)
```

Meaning:

```text
epsilon = 1.0   -> always random
epsilon = 0.1   -> 10% random, 90% greedy
epsilon = 0.01  -> 1% random, 99% greedy
```

The purpose is to explore heavily early, while the Q-network is still untrained, then gradually rely more on the learned Q-values.

#### Atari Epsilon Schedule

For Atari configs:

```python
exploration_schedule = PiecewiseSchedule(
    [
        (0, 1.0),
        (20000, 1),
        (total_steps / 2, 0.01),
    ],
    outside_value=0.01,
)
```

Example with `total_steps = 1_000_000`:

```text
step 0 to 20,000:       epsilon = 1.0
step 20,000 to 500,000: linearly decays from 1.0 to 0.01
after 500,000:          epsilon = 0.01
```

This schedule is a heuristic. It does not guarantee the Q-network is good by a certain step.

---

## 2. Training Loop

### 2.1 Data Flow

The core loop in `run_dqn.py` is:

```python
for step in range(total_steps):
    epsilon = exploration_schedule.value(step)
    action = agent.get_action(observation, epsilon)
    next_observation, reward, done, info = env.step(action)
    replay_buffer.insert(...)

    if done:
        reset_env_training()
    else:
        observation = next_observation

    if step >= learning_starts:
        batch = replay_buffer.sample(batch_size)
        update_info = agent.update(...)
```

Conceptually:

```text
current observation
-> choose action using epsilon-greedy Q-network
-> environment returns next observation and reward
-> store transition in replay buffer
-> sample random minibatch from replay
-> update critic network
```

The environment generates observations and rewards. The critic network does **not** generate observations. It only estimates Q-values.

---

### 2.2 Episodes vs Global Steps

The loop variable `step` is a global environment step count:

```python
for step in tqdm.trange(config["total_steps"]):
```

It is not an episode step count.

The first 1000 global steps can contain many episodes:

```text
global steps 0..123:    episode 1
global steps 124..301:  episode 2
global steps 302..477:  episode 3
...
```

Episodes end when:

```python
done == True
```

When an episode ends, the environment is reset:

```python
if done:
    reset_env_training()
else:
    observation = next_observation
```

If the episode is not done, the next observation becomes the current observation for the next loop iteration:

```text
obs_t -> action_t -> obs_{t+1}
obs_{t+1} becomes the next current observation
```

---

### 2.3 Episode Logging

The configs wrap environments with:

```python
RecordEpisodeStatistics(...)
```

At the end of an episode, this wrapper adds:

```python
info["episode"]["r"]  # total reward over the episode
info["episode"]["l"]  # episode length in environment steps
```

The training loop logs:

```python
logger.log({
    "Train_EpisodeReturn": info["episode"]["r"],
    "Train_EpisodeLen": info["episode"]["l"],
}, step)
```

These values are logged only when an episode ends.

---

### 2.4 Eval Log CSV Columns

Each run writes a local CSV file at:

```text
exp/<run_name>/log.csv
```

For example:

```text
exp/CartPole-v1_dqn_sd1_20260426_095149/log.csv
```

The same scalar metrics are also sent to Weights & Biases through `wandb.log`.

The CSV can contain different kinds of rows:

```text
evaluation rows       -> Eval_* columns are filled
episode-end rows      -> Train_EpisodeReturn and Train_EpisodeLen are filled
training-update rows  -> critic_loss, q_values, target_values, grad_norm, epsilon, lr are filled
```

Blank cells are normal. A row only fills the metrics that were logged at that specific `step`.

#### Eval Columns

Evaluation runs every `eval_interval` steps:

```python
if step % args.eval_interval == 0:
    trajectories = utils.sample_n_trajectories(
        eval_env,
        agent,
        args.num_eval_trajectories,
        ep_len,
    )
```

Then the code extracts returns and episode lengths:

```python
returns = [t["episode_statistics"]["r"] for t in trajectories]
ep_lens = [t["episode_statistics"]["l"] for t in trajectories]
```

These become:

```text
Eval_AverageReturn   mean return over evaluation episodes
Eval_StdReturn       standard deviation of evaluation returns
Eval_MaxReturn       best return among evaluation episodes
Eval_MinReturn       worst return among evaluation episodes
Eval_AverageEpLen    mean episode length over evaluation episodes
```

For `CartPole-v1`, reward is `+1` per timestep and the environment time limit is 500 steps. So:

```text
Eval_MaxReturn = 500
```

means at least one evaluation episode lasted the full 500 steps.

```text
Eval_AverageReturn = 500
```

means all evaluation episodes averaged the maximum possible return.

#### Training Episode Columns

When an online training episode ends, `RecordEpisodeStatistics` puts the episode statistics in `info["episode"]`, and the training loop logs:

```python
logger.log({
    "Train_EpisodeReturn": info["episode"]["r"],
    "Train_EpisodeLen": info["episode"]["l"],
}, step)
```

These columns mean:

```text
Train_EpisodeReturn   total reward from one training episode
Train_EpisodeLen      length of that training episode in environment steps
```

These are from the behavior policy used during training, so they include epsilon-greedy exploration. They can be lower than eval returns because training still sometimes takes random actions.

#### Training Update Columns

After `learning_starts`, the agent samples a replay batch and updates the critic:

```python
update_info = agent.update(...)
```

`DQNAgent.update_critic` returns:

```python
return {
    "critic_loss": loss.item(),
    "q_values": q_values.mean().item(),
    "target_values": target_values.mean().item(),
    "grad_norm": grad_norm.item(),
}
```

Then `run_dqn.py` adds:

```python
update_info["epsilon"] = epsilon
update_info["lr"] = agent.lr_scheduler.get_last_lr()[0]
```

These columns mean:

```text
critic_loss     mean squared TD error for the sampled replay batch
q_values        mean Q_current(obs, action) over the sampled batch
target_values   mean TD target over the sampled batch
grad_norm       gradient norm from the critic update
epsilon         random-action probability used for exploration at this step
lr              current optimizer learning rate
```

Training-update rows are logged every `log_interval` steps, except when the same step is also an eval step. At eval steps, the update metrics are merged into the eval row if training has already started:

```python
if step >= config["learning_starts"]:
    eval_metrics.update(update_info)
logger.log(eval_metrics, step)
```

#### Step Column

`step` is the global environment step from:

```python
for step in tqdm.trange(config["total_steps"]):
```

It is not an episode number. Multiple training episodes can finish before the next eval row appears.

#### How `log.csv` Is Saved

The logger is created in `make_logger`:

```python
logdir = os.path.join("exp", logdir)
os.makedirs(logdir, exist_ok=True)
return Logger(os.path.join(logdir, "log.csv"))
```

Every call to:

```python
logger.log(metrics, step)
```

adds `step` to the row, writes one CSV row, flushes the file, sends the same metrics to W&B, and stores a copy in memory:

```python
row["step"] = step
...
self.file.write(...)
self.file.flush()
wandb.log(row, step=step)
self.rows.append(copy.deepcopy(row))
```

If a later row introduces new metric names, the logger expands the CSV header and rewrites previous rows with blanks for columns that did not exist yet. That is why one `log.csv` can contain eval, training-episode, and update metrics in a single table.

#### Eval Is Not a Supervised Test Set

In supervised learning, data is often split into fixed train and test sets. In this RL code, data comes from interaction with Gym:

```text
observation -> DQN chooses action -> env returns next observation, reward, done
```

Training trajectories are collected with epsilon-greedy exploration, stored in replay, and used for updates. Eval trajectories are generated by running the current DQN policy in a separate eval environment, without inserting those transitions into replay or updating the network.

So evaluation answers:

```text
How well does the current policy perform when we run it in the environment?
```

It is not a held-out dataset in the supervised-learning sense. The eval environment is a separate instance, but it is still the same task, such as `CartPole-v1`.

Overfitting can still happen in RL, but it usually means overfitting to a narrow environment distribution, one random seed, specific initial states, or simulator details. For CartPole, this is less severe because the state is small and the task has a simple control rule, but one seed reaching return 500 is still weak evidence. A stronger check is to run several seeds and confirm that most runs reach high `Eval_AverageReturn` or `Eval_MaxReturn`.

---

## 3. Replay Buffer and Batches

### 3.1 Why Replay Buffer Exists

If DQN trained only on the newest transition every step, samples would be highly correlated:

```text
s_t, s_{t+1}, s_{t+2}, ...
```

Replay buffer reduces this problem by storing past transitions and sampling random minibatches:

```text
transition 500
transition 17
transition 920
transition 113
...
```

This makes updates more like supervised minibatch learning and improves stability.

The replay buffer stores transitions:

```text
(obs_t, action_t, reward_t, next_obs_t, done_t)
```

It does not store complete episodes as one batch. A sampled batch can mix transitions from many episodes and many different times.

---

### 3.2 Normal Replay Buffer

For state observations like CartPole:

```text
observation shape = (4,)
```

The normal `ReplayBuffer` stores full arrays:

```python
self.observations
self.actions
self.rewards
self.next_observations
self.dones
```

Example with `capacity = 1_000_000`:

```text
observations:      (1_000_000, 4)
actions:           (1_000_000,)
rewards:           (1_000_000,)
next_observations: (1_000_000, 4)
dones:             (1_000_000,)
```

Sampling with `batch_size = 128` gives:

```text
batch["observations"]:      (128, 4)
batch["actions"]:           (128,)
batch["rewards"]:           (128,)
batch["next_observations"]: (128, 4)
batch["dones"]:             (128,)
```

---

### 3.3 Memory-Efficient Replay Buffer

Atari observations are stacks of recent image frames:

```text
observation shape = (4, 84, 84)
```

Example:

```text
obs_t     = [f0, f1, f2, f3]
obs_{t+1} = [f1, f2, f3, f4]
```

A normal replay buffer would store both full stacks, duplicating `f1`, `f2`, and `f3`.

`MemoryEfficientReplayBuffer` instead stores individual frames once:

```text
framebuffer = [f0, f1, f2, f3, f4, ...]
```

Then each transition stores frame indices:

```text
observation_framebuffer_idcs[i]      = [0, 1, 2, 3]
next_observation_framebuffer_idcs[i] = [1, 2, 3, 4]
```

It stores:

```python
self.framebuffer
self.observation_framebuffer_idcs
self.next_observation_framebuffer_idcs
self.actions
self.rewards
self.dones
```

Example shapes:

```text
framebuffer:                        (2_000_000, 84, 84)
observation_framebuffer_idcs:       (1_000_000, 4)
next_observation_framebuffer_idcs:  (1_000_000, 4)
actions:                            (1_000_000,)
rewards:                            (1_000_000,)
dones:                              (1_000_000,)
```

When sampled, it reconstructs stacked observations:

```python
self.framebuffer[observation_framebuffer_idcs]
```

Sampling with `batch_size = 32` gives:

```text
batch["observations"]:      (32, 4, 84, 84)
batch["actions"]:           (32,)
batch["rewards"]:           (32,)
batch["next_observations"]: (32, 4, 84, 84)
batch["dones"]:             (32,)
```

---

### 3.4 Why Memory-Efficient Replay Needs Reset

Frame stacks must not cross episode boundaries.

If one episode ends with:

```text
[f97, f98, f99]
```

and a new episode starts with:

```text
g0
```

the first stack in the new episode should not be:

```text
[f97, f98, f99, g0]  # invalid
```

Instead it should repeat the first new frame:

```text
[g0, g0, g0, g0]
```

So on reset, the training loop calls:

```python
replay_buffer.on_reset(observation=observation[-1, ...])
```

This does **not** clear the replay buffer. It only tells the memory-efficient buffer:

```text
a new episode starts here; do not build frame histories from older frames
```

Old transitions remain available for training until overwritten by the ring buffer.

---

### 3.5 Learning Starts

The code does not train immediately:

```python
if step >= config["learning_starts"]:
    batch = replay_buffer.sample(config["batch_size"])
```

Before `learning_starts`, the agent only collects data.

Example:

```text
CartPole:     learning_starts = 1000
LunarLander:  learning_starts = 20000
Atari:        learning_starts = 20000
```

This avoids training on a tiny, highly correlated buffer at the beginning.

---

### 3.6 Batch Sampling

A replay batch is a random set of individual transitions, not an episode.

Example batch:

```text
episode 1, step 17
episode 5, step 3
episode 2, step 91
episode 5, step 4
episode 20, step 0
...
```

One episode may contribute only one transition to a batch. That is fine because one-step DQN only needs:

```text
(obs_t, action_t, reward_t, next_obs_t, done_t)
```

The target uses the immediate reward plus the estimated future value from `next_obs_t`.

---

### 3.7 Numpy to Torch Conversion

Replay buffers return NumPy arrays:

```python
batch = replay_buffer.sample(config["batch_size"])
```

Then:

```python
batch = ptu.from_numpy(batch)
```

converts every array in the dictionary to a PyTorch tensor on the configured device.

The keys and shapes stay the same:

```text
observations
actions
rewards
next_observations
dones
```

---

## 4. Critic Update

### 4.1 Prediction and Target

The core update is in `DQNAgent.update_critic`.

Input batch shapes:

```text
obs:      (B, *observation_shape)
action:   (B,)
reward:   (B,)
next_obs: (B, *observation_shape)
done:     (B,)
```

For Atari:

```text
obs:      (32, 4, 84, 84)
action:   (32,)
reward:   (32,)
next_obs: (32, 4, 84, 84)
done:     (32,)
```

For CartPole:

```text
obs:      (128, 4)
action:   (128,)
reward:   (128,)
next_obs: (128, 4)
done:     (128,)
```

#### Current Critic Prediction

The current critic predicts Q-values for all actions:

```python
qa_values = self.critic(obs)
```

Shape:

```text
(B, num_actions)
```

Then the code selects the Q-value for the action actually taken in the replay transition:

```python
q_values = torch.gather(
    qa_values,
    dim=1,
    index=action.long()[:, None],
).squeeze(1)
```

Shape:

```text
(B,)
```

This is:

```text
Q_current(obs_i, action_i)
```

#### TD Target

The target is:

```python
target_values = reward + discount * (1 - done.float()) * next_q_values
```

This corresponds to:

```text
y = r + gamma * max_a Q_target(next_obs, a)
```

If `done=True`, the future value is removed:

```text
y = r
```

because a true terminal state has no next-state value.

---

### 4.2 Why `torch.no_grad()` Is Used

The target computation is wrapped in:

```python
with torch.no_grad():
    ...
    target_values = ...
```

This tells PyTorch not to build an autograd graph through the target.

Gradients flow through:

```text
critic parameters -> qa_values -> gather -> q_values -> MSE loss
```

Gradients do not flow through:

```text
target_critic -> next_qa_values -> target_values
```

So the TD target is treated like a fixed supervised label for this update.

This is called a **semi-gradient TD update**. It uses gradient descent machinery, but not full gradient descent through the Bellman target.

---

### 4.3 Critic Loss

The critic loss is:

```python
self.critic_loss = nn.MSELoss()
```

and:

```python
loss = self.critic_loss(q_values, target_values)
```

This is supervised-learning-like:

```text
prediction = Q_current(obs, action_taken)
target     = reward + gamma * estimated future value
loss       = MSE(prediction, target)
```

The important difference from ordinary supervised learning is that the target is bootstrapped:

```text
real immediate reward
+ discounted value estimated by another network
```

The critic predicts expected return, not just immediate reward:

```text
Q(s, a) ~= r_t + gamma r_{t+1} + gamma^2 r_{t+2} + ...
```

---

### 4.4 Optimizer Step

After computing the loss:

```python
self.critic_optimizer.zero_grad()
loss.backward()
grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(...)
self.critic_optimizer.step()
self.lr_scheduler.step()
```

This does one gradient-based update to the current critic.

The target critic is not optimized directly by this optimizer. It is updated separately by copying parameters from the current critic.

---

## 5. Target Networks and Double DQN

### 5.1 Target Critic

DQN uses a target critic for stable bootstrapping:

```python
self.target_critic = make_critic(observation_shape, num_actions)
```

The target critic is a delayed copy of the current critic. It is used to compute TD targets:

```text
target = reward + gamma * Q_target(next_obs, next_action)
```

Why not use the current critic for both sides?

```text
Q_current(obs, action)
vs
reward + gamma * max_a Q_current(next_obs, a)
```

Because then the model would chase a target that changes every gradient step.

The target critic gives a more stable temporary label:

```text
fit current critic toward a slower-moving target
```

---

### 5.2 Hard Target Updates

This implementation uses hard target updates:

```python
def update_target_critic(self):
    self.target_critic.load_state_dict(self.critic.state_dict())
```

and:

```python
if step % self.target_update_period == 0:
    self.update_target_critic()
```

Meaning:

```text
every N steps:
    target_critic <- critic
```

This is classic DQN behavior.

It is not a soft/Polyak update. A soft update would blend parameters every step:

```text
target <- tau * target + (1 - tau) * critic
```

---

### 5.3 Standard DQN vs Double DQN

Standard DQN target:

```python
next_qa_values = self.target_critic(next_obs)
next_action = torch.argmax(next_qa_values, dim=1)
next_q_values = Q_target(next_obs, next_action)
```

The target critic both chooses and evaluates the next action:

```text
choose:   argmax_a Q_target(next_obs, a)
evaluate: Q_target(next_obs, chosen_action)
```

Double DQN target:

```python
next_action = torch.argmax(self.critic(next_obs), dim=1)
next_q_values = Q_target(next_obs, next_action)
```

The current critic chooses, and the target critic evaluates:

```text
choose:   argmax_a Q_current(next_obs, a)
evaluate: Q_target(next_obs, chosen_action)
```

---

### 5.4 Why Overestimation Happens

The `max` in the target can cause overestimation:

```text
max_a Q(next_obs, a)
```

Even if each action-value estimate has zero-mean noise, the max tends to select actions with positive noise.

Example:

```text
true values:
  a0 = 10
  a1 = 10
  a2 = 10

network estimates:
  a0 = 9.7
  a1 = 10.8
  a2 = 9.9
```

The max selects `10.8`, which is high because of noise, not because the action is truly better.

Standard DQN uses the same network values to select and evaluate the max action, so this upward bias can enter the target.

Double DQN reduces the bias by separating action selection and evaluation.

It does not fully eliminate overestimation because the current critic and target critic are still related; the target critic is a delayed copy of the current critic.

---

### 5.5 One-Step vs N-Step Targets

This implementation uses one-step targets:

```text
y_t = r_t + gamma * max_a Q(s_{t+1}, a)
```

An N-step target would be:

```text
y_t = r_t
    + gamma r_{t+1}
    + gamma^2 r_{t+2}
    + ...
    + gamma^{N-1} r_{t+N-1}
    + gamma^N max_a Q(s_{t+N}, a)
```

N-step targets can:

- propagate reward information faster
- reduce immediate reliance on inaccurate bootstrap values
- often improve early learning

But they also:

- increase variance
- are more sensitive to off-policy mismatch
- require the buffer/training loop to store or reconstruct multi-step sequences

N-step learning can be combined with Double DQN, but this project currently implements one-step DQN.

---

## 6. Algorithm Positioning

### 6.1 FQI vs Online Q-Learning vs This Code

#### Fitted Q-Iteration

Classic FQI:

```text
given dataset D
repeat:
    build targets using old Q
    solve regression problem over D
```

It is often written as:

```text
phi <- argmin_phi sum_i 1/2 * (Q_phi(s_i, a_i) - y_i)^2
```

#### Online Q-Learning

Online Q-learning:

```text
take action
observe one transition
take one TD update
```

With function approximation, the update is also based on squared TD error, but usually one sample or a small recent batch at a time.

#### This DQN Implementation

This code does:

```text
collect one environment step online
store it in replay
sample random minibatch from replay
take one gradient update
periodically update target network
```

So a precise description is:

```text
online DQN with replay-buffered fitted TD updates
```

It is similar to FQI because the loss is MSE regression to TD targets.

It is online because data collection and learning are interleaved every environment step.

---

### 6.2 Not Parallel or Async

This implementation is not synchronized parallel Q-learning and not asynchronous parallel Q-learning.

It has:

```text
one environment
one training loop
one current critic
one target critic
one replay buffer
```

There are no rollout workers, no parameter server, and no asynchronous gradient pushes.

Replay reduces correlation by random sampling, not by parallel data collection.

---

### 6.3 What `update()` Returns

`update_critic()` returns scalar diagnostics:

```python
return {
    "critic_loss": loss.item(),
    "q_values": q_values.mean().item(),
    "target_values": target_values.mean().item(),
    "grad_norm": grad_norm.item(),
}
```

These are used for logging:

```python
update_info = agent.update(...)
logger.log(update_info, step)
```

They are not additional training targets. They are just metrics to help inspect training.

---

## 7. End-to-End Summary

One DQN training step after `learning_starts` looks like:

```text
1. Use epsilon-greedy critic to choose an action.
2. Step the environment.
3. Store transition in replay.
4. Sample random minibatch from replay.
5. Current critic predicts Q(obs, all actions).
6. Gather Q(obs, action_taken).
7. Target critic estimates future value from next_obs.
8. Build TD target:
       reward + gamma * future_value
9. MSE loss:
       Q_current(obs, action_taken) vs TD target
10. Backprop only through current critic prediction.
11. Optimizer updates current critic.
12. Every target_update_period steps, copy current critic to target critic.
```

The core idea:

```text
learn a Q-network whose values make observed transitions Bellman-consistent,
while using replay and a target network to make neural-network bootstrapping stable enough to train.
```
