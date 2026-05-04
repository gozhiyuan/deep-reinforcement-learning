# RL Algorithm Comparison Guide

This guide compares the three algorithm families used across HW2 and HW3:

- policy gradient / actor-critic from HW2
- DQN from HW3
- SAC from HW3

The main goal is to make clear what each method learns, what data it uses, how actions are chosen, and why different environments use different algorithms.

---

## 1. Short Summary

```text
Policy gradient:
  directly learns a policy pi(a | s)
  usually on-policy
  works for discrete or continuous actions

DQN:
  learns Q-values for discrete actions
  off-policy with replay buffer
  no explicit actor network

SAC:
  learns both actor pi(a | s) and critic Q(s, a)
  off-policy with replay buffer
  designed here for continuous actions
```

One-line intuition:

```text
Policy gradient:
  learn from actions the current policy just took

DQN:
  learn Q-values, then choose the highest-Q discrete action

SAC:
  learn Q-values, then train an actor to produce high-Q continuous actions
```

---

## 2. Main Comparison Table

| Method | Learns Policy? | Learns Critic? | Data | Best Action Space |
|---|---:|---:|---|---|
| Policy Gradient / Actor-Critic | Yes | Optional `V(s)` | Fresh rollout batch | Discrete or continuous |
| DQN | Implicit only | `Q(s, a)` | Replay buffer | Discrete |
| SAC | Yes | `Q(s, a)` | Replay buffer | Continuous |

More detail:

| Topic | Policy Gradient / Actor-Critic | DQN | SAC |
|---|---|---|---|
| Policy | explicit actor `pi(a|s)` | no actor, `argmax Q` | explicit stochastic actor |
| Critic output | usually `V(s)` | Q-value for each discrete action | Q-value for one `(s,a)` pair |
| Actor update | `log_prob(action) * advantage` | none | maximize `Q + entropy` |
| Exploration | policy sampling | epsilon-greedy | stochastic actor + entropy |
| Replay buffer | no, usually | yes | yes |
| On/off policy | on-policy | off-policy | off-policy |
| Sample efficiency | lower | higher | higher |

---

## 3. Policy Gradient / Actor-Critic

HW2 policy gradient directly learns a policy:

```text
actor(obs) -> action distribution
```

For discrete actions:

```text
obs -> logits -> Categorical distribution -> action id
```

For continuous actions:

```text
obs -> mean/std -> Gaussian distribution -> action vector
```

### 3.1 Data Flow

HW2 uses fresh rollout data:

```text
collect trajectories with current policy
  -> compute returns / advantages
  -> update actor and maybe critic
  -> discard data
  -> collect new trajectories
```

This is **on-policy**:

```text
the update data must come from the current policy
```

### 3.2 Actor Update

Policy-gradient actor loss uses the actions that were actually taken in rollout:

```text
loss = -log pi(old_action | obs) * advantage
```

Meaning:

```text
positive advantage:
  make old_action more likely

negative advantage:
  make old_action less likely
```

### 3.3 Critic

The HW2 actor-critic critic usually estimates:

```text
V(s)
```

It answers:

```text
how good is this state?
```

Then advantage can be:

```text
A(s, a) = return - V(s)
```

The critic is mainly a baseline to reduce policy-gradient variance.

---

## 4. DQN

DQN is value-based. It does not use a separate actor network.

The critic maps:

```text
obs -> Q-value for each discrete action
```

Example CartPole:

```text
obs.shape = (batch_size, 4)
q_values.shape = (batch_size, 2)

q_values[i] = [Q(obs_i, left), Q(obs_i, right)]
```

Action selection:

```text
action = argmax_a Q(obs, a)
```

Exploration:

```text
epsilon-greedy
```

### 4.1 Data Flow

DQN is off-policy:

```text
env step
  -> store transition in replay buffer
  -> sample random minibatch from replay
  -> update critic
```

Replay transition:

```text
(obs, action, reward, next_obs, done)
```

Example CartPole batch:

```text
observations:      (128, 4)
actions:           (128,)
rewards:           (128,)
next_observations: (128, 4)
dones:             (128,)
```

### 4.2 Critic Update

DQN target:

```text
target = reward + gamma * (1 - done) * max_a' Q_target(next_obs, a')
```

Loss:

```text
loss = MSE(Q(obs, replay_action), target)
```

Why discrete only:

```text
DQN needs max over actions.
For discrete actions, evaluate all actions and take max.
For continuous actions, max over infinitely many actions is hard.
```

---

## 5. SAC

SAC is an off-policy actor-critic method for continuous actions.

It learns:

```text
actor:
  pi(a | s)

critic:
  Q(s, a)
```

The actor outputs a distribution:

```text
obs -> mean/std -> tanh-squashed Gaussian -> action vector
```

The critic scores one state-action pair:

```text
critic(obs, action) -> Q(obs, action)
```

Example Hopper:

```text
obs.shape = (batch_size, 11)
action.shape = (batch_size, 3)
q_values.shape = (num_critics, batch_size)
```

### 5.1 Data Flow

SAC also uses replay:

```text
env step
  -> store transition in replay buffer
  -> sample replay batch
  -> update critic
  -> update actor
  -> update target critic
```

Replay batch:

```text
obs, action, reward, next_obs, done
```

### 5.2 Critic Update

SAC critic update is Bellman regression:

```text
Q(obs, replay_action) -> target
```

Target:

```text
next_action ~ actor(next_obs)

target =
  reward
  + gamma * (1 - done)
    * [Q_target(next_obs, next_action) + temperature * entropy]
```

Key difference from DQN:

```text
DQN:
  future action = argmax over discrete Q-values

SAC:
  future action = sample from actor(next_obs)
```

### 5.3 Actor Update

SAC actor update does not imitate replay actions.

It uses replay observations but samples new actions from the current actor:

```text
action = actor(obs).rsample()
q = critic(obs, action)
```

Actor loss:

```text
loss = -Q(obs, actor_action) - temperature * entropy
```

Minimizing that means maximizing:

```text
Q(obs, actor_action) + temperature * entropy
```

So SAC asks:

```text
what action should my current actor produce here?
```

not:

```text
what action did the replay buffer contain?
```

---

## 6. On-Policy vs Off-Policy

This is the biggest conceptual split.

### 6.1 On-Policy: HW2 Policy Gradient

```text
collect data with current policy
update using that data
discard data
collect fresh data
```

Why:

```text
policy-gradient update uses log pi(action | obs)
for actions sampled by the current policy
```

### 6.2 Off-Policy: DQN and SAC

```text
store transitions in replay
sample old transitions
update many times from old data
```

Why possible:

```text
DQN:
  learns Bellman Q-values from transitions

SAC:
  learns Q(s,a) from transitions
  actor can be updated using replay states and current critic
```

Off-policy methods are usually more sample efficient because they reuse data.

---

## 7. Action Space Fit

### 7.1 Discrete Actions

Examples:

```text
CartPole:
  action = 0 or 1

Atari:
  action = integer button/action id
```

Good methods:

```text
DQN
PPO / policy gradient
```

DQN works naturally because:

```text
critic(obs) outputs one Q-value per action
```

### 7.2 Continuous Actions

Examples:

```text
InvertedPendulum:
  action.shape = (1,)

Hopper:
  action.shape = (3,)

HalfCheetah:
  action.shape = (6,)
```

Good methods:

```text
SAC
PPO / policy gradient
TD3 / DDPG
```

SAC works naturally because:

```text
actor directly outputs continuous action distribution
```

DQN is not natural because:

```text
max over all continuous actions is hard
```

---

## 8. Exploration

Policy gradient:

```text
exploration comes from sampling the policy
optional entropy bonus may help
```

DQN:

```text
epsilon-greedy
with probability epsilon, take random action
otherwise, take argmax Q action
```

SAC:

```text
actor is stochastic
entropy is part of the objective
temperature controls randomness
```

SAC entropy term:

```text
maximize Q + temperature * entropy
```

---

## 9. Real-World Usage

### 9.1 Policy Gradient / PPO

PPO-style policy gradient is widely used because it is stable, simple, and works for discrete or continuous actions.

Common uses:

```text
large-scale policy optimization
games
robotics simulation
RLHF-style training
research baselines
```

### 9.2 DQN

DQN is important for discrete-action value learning.

Common uses:

```text
Atari-style tasks
simple discrete-control tasks
discrete decision problems
```

It is less natural for robotics torque control because actions are continuous.

### 9.3 SAC

SAC is common for sample-efficient continuous control.

Common uses:

```text
robot arms
locomotion
MuJoCo benchmarks
sim-to-real robot learning
continuous autonomous control
```

SAC is attractive when environment interaction is expensive because replay improves sample efficiency.

---

## 10. Rule of Thumb

```text
Discrete action task:
  DQN or PPO

Continuous action task:
  SAC or PPO

Need high sample efficiency:
  SAC for continuous actions
  DQN for discrete actions

Need simple stable scaling:
  PPO-style policy gradient

Need direct continuous control with replay:
  SAC
```

---

## 11. Mental Model

Policy gradient:

```text
"The action I took worked better/worse than expected,
so make it more/less likely next time."
```

DQN:

```text
"Learn the value of each discrete action,
then choose the action with the highest value."
```

SAC:

```text
"Learn a Q-function from replay,
then train a stochastic actor to produce high-Q actions
while keeping enough entropy for exploration."
```

