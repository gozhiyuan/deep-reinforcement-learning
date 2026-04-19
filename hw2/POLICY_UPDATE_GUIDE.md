# HW2 Policy Update Guide

This note explains what happens after rollout data has been collected.

It focuses on:

- rewards, returns, and reward-to-go
- how trajectory lists become one flat update batch
- what advantages are
- how the critic baseline is used and trained
- how GAE computes advantages with TD errors
- why the policy network runs again during update
- how `log_prob`, `actions.long()`, and the policy-gradient loss work

For the rollout and policy-network shape basics, see `DATA_AND_POLICY_NETWORK_GUIDE.md`.

---

## 1. Big Picture

The training loop in `src/scripts/run.py` does this repeatedly:

```text
collect trajectories with current policy
    -> compute returns / advantages from rewards
    -> update policy weights
    -> collect next batch with updated policy
```

The policy is not updated after every single environment step. It is updated after a batch of rollout data is collected.

Example:

```text
batch 0: rollout with policy weights theta_0 -> update -> theta_1
batch 1: rollout with policy weights theta_1 -> update -> theta_2
batch 2: rollout with policy weights theta_2 -> update -> theta_3
```

This is online, on-policy reinforcement learning:

```text
online     -> the agent collects new environment data during training
on-policy  -> each batch is collected by the current policy, then used for that update
```

The batch is not kept in a replay buffer. After the actor/critic update, the next iteration collects fresh rollouts with the updated policy.

### 1.1 `PGAgent.update(...)` Diagram

Inside `PGAgent.update(...)`, the collected rollout batch goes through:

```text
inputs from run.py:
obs, actions, rewards, terminals
    each is a list of per-trajectory arrays

        |
        v

1. _calculate_q_vals(rewards)
    rewards are still split by trajectory
    q_values: list of per-trajectory arrays

        |
        v

2. concatenate all trajectories
    obs        -> (B, ob_dim)
    actions    -> (B,) or (B, ac_dim)
    rewards    -> (B,)
    terminals  -> (B,)
    q_values   -> (B,)

        |
        v

3. _estimate_advantage(obs, rewards, q_values, terminals)
    no critic:      advantages = q_values
    critic:         advantages = q_values - V(s)
    critic + GAE:   advantages from TD-error recursion

        |
        v

4. actor.update(obs, actions, advantages)
    rerun policy on old obs
    compute log_prob(old actions)
    update actor weights

        |
        v

5. if critic exists: critic.update(obs, q_values)
    train V(s) to predict Monte Carlo q_values
```

Here:

```text
B = total timesteps in the collected batch
```

---

## 2. Batch Data Before `PGAgent.update(...)`

After rollout, `run.py` reorganizes trajectories:

```python
trajs_dict = {k: [traj[k] for traj in trajs] for k in trajs[0]}
```

Example for CartPole:

```text
ob_dim = 4

trajectory 0 length T_0 = 3
trajectory 1 length T_1 = 5
```

Then:

```python
trajs_dict["observation"] = [
    obs_0,  # shape (3, 4)
    obs_1,  # shape (5, 4)
]

trajs_dict["action"] = [
    acs_0,  # shape (3,)
    acs_1,  # shape (5,)
]

trajs_dict["reward"] = [
    rew_0,  # shape (3,)
    rew_1,  # shape (5,)
]
```

The first dimension is trajectory length:

```text
obs_0.shape = (3, 4)
               ^  ^
               |  |
               |  ob_dim = 4 CartPole state numbers
               T_0 = 3 environment steps
```

Within one trajectory, timestep indices match:

```text
obs_0[0] -> acs_0[0] -> rew_0[0]
obs_0[1] -> acs_0[1] -> rew_0[1]
obs_0[2] -> acs_0[2] -> rew_0[2]
```

---

## 3. Rewards

Rewards come from the environment:

```python
next_ob, rew, done, info = env.step(ac)
rewards.append(rew)
```

There is one reward per environment step.

For CartPole, the standard reward is usually:

```text
+1.0 for each timestep before the rollout ends
```

So if CartPole survives 3 steps:

```python
rewards = np.array([1.0, 1.0, 1.0], dtype=np.float32)
```

No negative reward is needed. Failing early is bad because the agent stops collecting future `+1` rewards.

---

## 4. Q-Values From Rewards

In `PGAgent.update(...)`:

```python
q_values = self._calculate_q_vals(rewards)
```

This happens before concatenation, while `rewards` is still a list of trajectories:

```python
rewards = [
    np.array([1.0, 1.0, 1.0]),             # trajectory 0
    np.array([1.0, 1.0, 1.0, 1.0, 1.0]),   # trajectory 1
]
```

This matters because returns should not cross trajectory boundaries.

### 4.1 Discounted Return

If `use_reward_to_go=False`, every timestep in one trajectory gets the same full trajectory return.

Example:

```text
rewards = [1, 1, 1]
gamma = 0.9

discounts = [1, 0.9, 0.81]
discounted_return = 1 + 0.9 + 0.81 = 2.71
```

The returned Q-values are:

```python
q_values = np.array([2.71, 2.71, 2.71])
```

Why repeat the same number?

Because the policy update needs one `q_value` for each `(observation, action)` pair:

```text
obs[0], action[0] -> q_values[0]
obs[1], action[1] -> q_values[1]
obs[2], action[2] -> q_values[2]
```

### 4.2 Reward-To-Go

If `use_reward_to_go=True`, each timestep only gets future rewards from that timestep onward.

Example:

```text
rewards = [1, 1, 1]
gamma = 0.9
```

Work backward:

```text
t=2 -> 1
t=1 -> 1 + 0.9 * 1 = 1.9
t=0 -> 1 + 0.9 * 1.9 = 2.71
```

The returned Q-values are:

```python
q_values = np.array([2.71, 1.9, 1.0])
```

Reward-to-go is usually less noisy because later actions are not credited for rewards that happened before them.

---

## 5. Concatenation

After Q-values are computed per trajectory, `PGAgent.update(...)` flattens the batch:

```python
obs = np.concatenate(obs)
actions = np.concatenate(actions)
rewards = np.concatenate(rewards)
terminals = np.concatenate(terminals)
q_values = np.concatenate(q_values)
```

Example before concat:

```text
obs = [
    array shape (3, 4),
    array shape (5, 4),
]

actions = [
    array shape (3,),
    array shape (5,),
]
```

After concat:

```text
batch_size = 3 + 5 = 8

obs.shape       = (8, 4)
actions.shape   = (8,)
rewards.shape   = (8,)
q_values.shape  = (8,)
terminals.shape = (8,)
```

The policy update no longer needs explicit trajectory boundaries. It only needs aligned rows:

```text
obs[i], actions[i], advantages[i]
```

For GAE, boundaries still matter, so `terminals` marks the final step of each trajectory.

Example:

```python
terminals = np.array([0, 0, 1, 0, 0, 0, 0, 1])
```

---

## 6. Advantages

The actor update does not use raw rewards directly. It uses `advantages`:

```python
advantages = self._estimate_advantage(obs, rewards, q_values, terminals)
```

Advantage means:

```text
how good this sampled action was compared with what was expected
```

Input shapes after concatenation:

```text
obs.shape       = (B, ob_dim)
rewards.shape   = (B,)
q_values.shape  = (B,)
terminals.shape = (B,)
```

Output shape:

```text
advantages.shape = (B,)
```

### 6.1 No Baseline

If there is no critic baseline:

```python
advantages = q_values.copy()
```

So the advantage is just the Monte Carlo return estimate.

Example:

```python
q_values = np.array([2.71, 1.9, 1.0])
advantages = np.array([2.71, 1.9, 1.0])
```

`q_values.copy()` makes `advantages` a separate NumPy array. In this code `q_values` is not modified later, but the copy prevents accidental in-place changes to `q_values` if advantage logic changes later.

### 6.2 With Baseline

If there is a critic:

```python
values = self.critic(obs)
advantages = q_values - values
```

The critic is the value network in `src/networks/critics.py`:

```python
class ValueCritic(nn.Module):
    def forward(self, obs):
        return self.network(obs).squeeze(-1)
```

It maps:

```text
obs.shape    = (B, ob_dim)
values.shape = (B,)
```

Each `values[i]` is:

```text
V(s_i) = critic estimate of expected return from obs[i]
```

Example:

```python
q_values = np.array([2.71, 1.9, 1.0])
values   = np.array([2.00, 2.10, 0.75])

advantages = np.array([0.71, -0.20, 0.25])
```

Meaning:

```text
0.71   better than expected
-0.20  worse than expected
0.25   slightly better than expected
```

This reduces variance. Instead of saying:

```text
action was good because return was 2.71
```

it says:

```text
action was good because return was 0.71 better than expected
```

### 6.3 How The Critic Is Trained

The critic is trained later in `PGAgent.update(...)`:

```python
for _ in range(self.baseline_gradient_steps):
    critic_info = self.critic.update(obs, q_values)
```

Inside `ValueCritic.update(...)`:

```python
loss = F.mse_loss(self(obs), q_values)
```

So the critic learns:

```text
V(s_i) should predict q_values[i]
```

Shape example:

```text
obs.shape      = (8, 4)
q_values.shape = (8,)
self(obs)      = (8,)
loss           = scalar
```

Important timing:

```text
1. current critic computes values for advantages
2. actor updates using those advantages
3. critic then trains to better predict q_values
```

### 6.4 GAE: What `gamma` And `delta` Mean

GAE runs only when both are true:

```text
critic exists
gae_lambda is not None
```

Example command:

```bash
uv run src/scripts/run.py --use_baseline --gae_lambda 0.95
```

`gamma` is the discount factor:

```python
parser.add_argument("--discount", type=float, default=1.0)
```

Meaning:

```text
gamma = 1.0   future rewards count fully
gamma = 0.99  future rewards count slightly less
gamma = 0.9   future rewards decay faster
```

`delta` is the one-step TD error:

```text
delta[i] = rewards[i] + gamma * values[i + 1] - values[i]
```

In code:

```python
delta = (
    rewards[i]
    + self.gamma * values[i + 1] * next_nonterminal
    - values[i]
)
```

Mapping:

```text
rewards[i]    = r_i
values[i]     = V(s_i)
values[i + 1] = V(s_{i+1})
self.gamma    = gamma
```

Example:

```text
reward = 1.0
gamma = 0.9
V(s_i) = 2.0
V(s_{i+1}) = 1.5

delta = 1.0 + 0.9 * 1.5 - 2.0 = 0.35
```

Meaning:

```text
this step was 0.35 better than the critic expected
```

### 6.5 Why GAE Adds Future Advantages

One-step `delta[i]` only checks:

```text
did this immediate transition beat V(s_i)?
```

But action `a_i` can affect many future rewards. So GAE uses:

```text
advantage[i] = delta[i] + gamma * lambda * advantage[i + 1]
```

Meaning:

```text
current advantage
= this step's surprise
+ discounted future surprises
```

If the current action sets up a good future, `advantage[i + 1]` passes some credit backward.

If the current action sets up a bad future, `advantage[i + 1]` passes some blame backward.

`lambda` controls how far that credit/blame travels:

```text
lambda = 0
    advantage[i] = delta[i]
    low variance, more bias

lambda close to 1
    uses more future deltas
    closer to Monte Carlo advantage
    less bias, more variance
```

### 6.6 GAE Code And Example

The code appends dummy values:

```python
values = np.append(values, [0])
advantages = np.zeros(batch_size + 1)
```

Why?

```text
values[i + 1] and advantages[i + 1] are always valid, even at the final index
```

Then it loops backward:

```python
for i in reversed(range(batch_size)):
    next_nonterminal = 1 - terminals[i]
    delta = rewards[i] + gamma * values[i + 1] * next_nonterminal - values[i]
    advantages[i] = delta + gamma * lambda * next_nonterminal * advantages[i + 1]
```

`next_nonterminal` prevents crossing trajectory boundaries:

```text
terminal[i] = 0 -> next_nonterminal = 1 -> use next value/advantage
terminal[i] = 1 -> next_nonterminal = 0 -> stop recursion here
```

Small one-trajectory example:

```text
rewards   = [1, 1, 1]
values    = [2.0, 1.5, 0.5]
terminals = [0, 0, 1]
gamma = 0.9
lambda = 0.95
```

Append dummy:

```text
values     = [2.0, 1.5, 0.5, 0.0]
advantages = [0.0, 0.0, 0.0, 0.0]
```

Backward recursion:

```text
i=2:
next_nonterminal = 0
delta = 1 + 0 - 0.5 = 0.5
adv[2] = 0.5

i=1:
next_nonterminal = 1
delta = 1 + 0.9 * 0.5 - 1.5 = -0.05
adv[1] = -0.05 + 0.9 * 0.95 * 0.5 = 0.3775

i=0:
next_nonterminal = 1
delta = 1 + 0.9 * 1.5 - 2.0 = 0.35
adv[0] = 0.35 + 0.9 * 0.95 * 0.3775 = 0.6727625
```

Remove dummy:

```python
advantages = advantages[:-1]
```

Final:

```text
advantages = [0.6727625, 0.3775, 0.5]
```

### 6.7 GAE With Multiple Trajectories

After concat, two trajectories might look like:

```text
trajectory 0 length 3
trajectory 1 length 5

terminals = [0, 0, 1, 0, 0, 0, 0, 1]
```

At index `2`:

```text
terminals[2] = 1
next_nonterminal = 0
```

So the GAE recursion does not use index `3`, which belongs to the next trajectory.

### 6.8 Normalize Advantages

If enabled:

```python
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
```

Shape does not change:

```text
before: advantages.shape = (B,)
after:  advantages.shape = (B,)
```

Example:

```text
advantages = [2, 4, 6]
mean = 4
std = 1.633

normalized = [-1.225, 0, 1.225]
```

This makes actor updates more stable.

### 6.9 Complexity

The GAE loop is linear in the number of collected timesteps:

```text
time:   O(B)
memory: O(B)
```

where:

```text
B = total timesteps after concatenation
```

If using a critic, the critic forward pass also costs:

```text
O(B * neural_network_forward_cost)
```

---

## 7. Actor Update

The actor update is called from `PGAgent.update(...)`:

```python
info = self.actor.update(obs, actions, advantages)
```

At this point:

```text
obs.shape        = (B, ob_dim)
actions.shape    = (B,) for discrete actions
actions.shape    = (B, ac_dim) for continuous actions
advantages.shape = (B,)
```

For CartPole with `B=8`:

```text
obs.shape        = (8, 4)
actions.shape    = (8,)
advantages.shape = (8,)
```

### 7.1 NumPy To Torch

Inside `MLPPolicyPG.update(...)`:

```python
obs = ptu.from_numpy(obs)
actions = ptu.from_numpy(actions)
advantages = ptu.from_numpy(advantages)
```

`ptu.from_numpy(...)` converts NumPy arrays to PyTorch tensors and moves them to `ptu.device`.

`ptu.device` is either:

```text
cpu
cuda:0
```

depending on `ptu.init_gpu(...)`.

### 7.2 Running The Policy Again

The update runs:

```python
action_distribution = self.forward(obs)
```

This is not another environment rollout.

No `env.step(...)` happens here.

This only recomputes the current policy distribution for the old collected observations.

Example:

```text
obs.shape = (8, 4)

self.forward(obs)
    -> 8 action distributions
```

For CartPole:

```text
logits.shape = (8, 2)
```

because CartPole has two actions:

```text
0 = left
1 = right
```

### 7.3 Why Not Store Rollout Logits?

During rollout:

```python
@torch.no_grad()
def get_action(...):
```

This means PyTorch does not save the gradient graph.

The rollout is for acting:

```text
obs -> policy -> sample action -> env.step(action)
```

The update is for learning:

```text
old obs -> policy -> log_prob(old action) -> loss -> backward
```

Rerunning the policy during update builds the fresh graph needed for:

```python
loss.backward()
```

PPO-style algorithms sometimes store old log-probs from rollout, but they still rerun the current policy during update to get new log-probs with gradients.

---

## 8. `actions.long()` And `log_prob(...)`

For discrete actions:

```python
actions = actions.long()
```

CartPole example:

```python
actions = torch.tensor([1.0, 0.0, 1.0])
actions.long()
# torch.tensor([1, 0, 1])
```

`Categorical.log_prob(...)` expects integer action IDs.

Then:

```python
log_probs = action_distribution.log_prob(actions)
```

Example distributions:

```text
actions = [1, 0, 1]

obs[0] -> [left=0.4, right=0.6]
obs[1] -> [left=0.7, right=0.3]
obs[2] -> [left=0.2, right=0.8]
```

Then:

```text
log_probs[0] = log(0.6)
log_probs[1] = log(0.7)
log_probs[2] = log(0.8)
```

So `log_prob` asks:

```text
Under the current policy, how likely was the action we actually sampled?
```

It does not sample a new action.

---

## 9. Policy-Gradient Loss

The loss is:

```python
loss = -(log_probs * advantages).mean()
```

Example:

```python
log_probs = np.array([np.log(0.6), np.log(0.7), np.log(0.8)])
advantages = np.array([2.0, -1.0, 3.0])
```

Meaning:

```text
action 0 had positive advantage -> make it more likely
action 1 had negative advantage -> make it less likely
action 2 had positive advantage -> make it more likely
```

The negative sign is there because PyTorch minimizes losses, but policy gradient wants to maximize:

```python
(log_probs * advantages).mean()
```

So the code minimizes:

```python
-((log_probs * advantages).mean())
```

`.mean()` turns the per-timestep values into one scalar loss.

It also keeps gradient size more stable when the batch size changes.

---

## 10. Weight Update

The policy weights change here:

```python
self.optimizer.zero_grad()
loss.backward()
self.optimizer.step()
```

Meaning:

```text
zero_grad()       clear old gradients
loss.backward()   compute new gradients
step()            update neural-network weights
```

After `optimizer.step()`, the same observation may produce different action probabilities.

Example:

```text
before update:
obs[0] -> [left=0.4, right=0.6]

sampled action = right
advantage = +2.0

after update:
obs[0] -> [left=0.3, right=0.7]
```

The next rollout uses the updated policy.

---

## 11. Supervised Learning Comparison

The update looks like supervised learning because it has:

```text
inputs -> network -> loss -> backward -> optimizer step
```

But the target is different.

Supervised learning:

```text
"this is the correct label"
```

Policy gradient:

```text
"this action was sampled, and the reward-based advantage says it was good or bad"
```

So:

```text
positive advantage -> increase probability of sampled action
negative advantage -> decrease probability of sampled action
near-zero advantage -> small change
```

---

## 12. Summary

One training iteration is:

```text
1. Roll out current policy.
2. Store obs, actions, rewards, terminals.
3. Compute Q-values from actual sampled rewards while trajectories are separate.
4. Concatenate trajectories into one flat batch.
5. Compute advantages:
   - no critic: advantages = q_values
   - critic: advantages = q_values - V(s)
   - critic + GAE: advantages from TD-error recursion
6. Run policy on old observations again.
7. Compute log-probability of old sampled actions.
8. Weight log-probs by advantages.
9. Backprop the negative average actor loss.
10. If critic exists, train critic with MSE against q_values.
11. Next batch uses updated policy weights.
```

Shape summary for CartPole:

```text
Before concat:
obs       [array (3, 4), array (5, 4)]
actions   [array (3,),  array (5,)]
rewards   [array (3,),  array (5,)]

After concat:
obs        (8, 4)
actions    (8,)
rewards    (8,)
q_values   (8,)
advantages (8,)

Policy output:
logits      (8, 2)
log_probs   (8,)
actor loss  scalar

Critic output if using baseline:
values       (8,)
critic loss  scalar
```
