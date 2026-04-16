# Train and Eval Process Notes

This note summarizes how the imitation learning code in `hw1/src/hw1_imitation` works, with emphasis on the training loop, evaluation rollout, tensor shapes, normalization, logging, and a few PyTorch details. It covers both supported policies:

- `MSEPolicy`
- `FlowMatchingPolicy`

## 1. Overview

The homework uses **behavior cloning / imitation learning** on the Push-T task:

- Training data contains expert trajectories.
- The policy learns to map a **current state** to a **chunk of future actions**.
- At evaluation time, the learned policy is rolled out in the real Push-T simulator.

There are two distinct phases:

1. **Training**
   - Supervised learning on a fixed dataset.
   - No environment interaction is needed.

2. **Evaluation**
   - Run the learned policy inside the Gym Push-T environment.
   - The environment returns the next state, reward, and termination flags.

## 2. Environment and Data

### 2.1 Push-T environment

The environment is provided by the installed `gym-pusht` package:

```python
env = gym.make("gym_pusht/PushT-v0", obs_type="state", render_mode="rgb_array")
```

This is a prebuilt simulator. We do not manually compute next states.

#### Action format

The environment action space is:

```text
Box(0.0, 512.0, (2,), float32)
```

So one action has shape `(2,)` and means:

```text
[target_x, target_y]
```

This is not a direct next-state command. It is a 2D target position for the agent. The environment uses its own controller and physics to move the agent and block.

#### State format

With `obs_type="state"`, the environment state has shape `(5,)`:

```text
[agent_x, agent_y, block_x, block_y, block_angle]
```

So in this homework:

- `state_dim = 5`
- `action_dim = 2`

### 2.2 Dataset structure

The offline dataset contains:

- `states`: shape `(N, 5)`
- `actions`: shape `(N, 2)`

The custom dataset `PushtChunkDataset` turns this into supervised samples:

```text
state_t -> [action_t, action_{t+1}, ..., action_{t+chunk_size-1}]
```

So one training sample is:

- `state`: shape `(5,)`
- `action_chunk`: shape `(chunk_size, 2)`

After batching with the DataLoader:

- `state`: shape `(batch_size, 5)`
- `action_chunk`: shape `(batch_size, chunk_size, 2)`

Important: training samples are built from overlapping windows in the expert dataset. We are not rolling the model forward during training.

### 2.3 Normalization

The code computes a `Normalizer` from the training data:

```python
state_mean, state_std
action_mean, action_std
```

It uses standard score normalization:

```text
normalized = (value - mean) / std
```

#### Why normalize?

- Positions are on a scale around `[0, 512]`.
- Different dimensions can have different magnitudes.
- Training is usually easier when inputs and targets are roughly zero-mean and unit-scale.

#### Training

The dataset normalizes:

- states before feeding them to the model
- action chunks before computing loss

So the model learns in normalized coordinates.

For the two policies, that means:

- MSE policy: normalized state -> normalized clean action chunk
- flow policy: normalized state + noisy normalized action chunk + flow timestep -> normalized velocity tensor

#### Evaluation

At eval time:

1. raw env state is normalized before inference
2. model predicts normalized actions
3. predicted actions are denormalized back to the environment scale

Denormalization is:

```text
action = normalized_action * action_std + action_mean
```

This is necessary because the environment expects real Push-T coordinates, not normalized values near zero.

#### Do we keep the training mean/std?

Yes. We fit normalization on the training data and reuse the same statistics at evaluation/test time. That is the standard approach.

If future data comes from a very different range, performance may degrade. That is a distribution-shift problem, not something to solve by recomputing statistics on test data.

## 3. Policies and Losses

The code supports two policies with the same outer training loop but different internal prediction targets.

### 3.1 MSE policy

The MSE policy is an MLP that predicts an entire clean action chunk at once.

#### Input and output shapes

Input to the network:

```text
(batch_size, state_dim)
```

Raw output from the last linear layer:

```text
(batch_size, chunk_size * action_dim)
```

This output is flat, so it is reshaped with:

```python
pred.view(-1, self.chunk_size, self.action_dim)
```

The reshaped output becomes:

```text
(batch_size, chunk_size, action_dim)
```

Example: if `batch_size=128`, `chunk_size=8`, and `action_dim=2`:

- before reshape: `(128, 16)`
- after reshape: `(128, 8, 2)`

The `-1` tells PyTorch to infer the batch dimension automatically.

#### Loss

The loss is computed with:

```python
nn.functional.mse_loss(pred_chunk, action_chunk)
```

This applies elementwise mean squared error over the whole tensor.

For one scalar:

```text
(pred - target)^2
```

For one 2D action:

```text
((x_pred - x_true)^2 + (y_pred - y_true)^2) / 2
```

This is **not** Euclidean distance.

Euclidean distance would be:

```text
sqrt((x_pred - x_true)^2 + (y_pred - y_true)^2)
```

In the actual code, MSE is averaged over:

- batch dimension
- chunk dimension
- action dimension

So the final batch loss is the mean squared error over all numbers in:

```text
(batch_size, chunk_size, action_dim)
```

### 3.2 Flow matching policy

The flow policy predicts a **velocity tensor** instead of a clean chunk directly.

Its MLP input contains:

- the current normalized state
- the current noisy or interpolated action chunk
- the flow timestep `τ`

The implementation flattens the chunk and concatenates:

```text
[state, flattened_noisy_chunk, tau]
```

So the MLP input shape is:

```text
(batch_size, state_dim + chunk_size * action_dim + 1)
```

and the MLP output is reshaped to:

```text
(batch_size, chunk_size, action_dim)
```

That output is interpreted as the predicted velocity:

```text
v_θ(o_t, A_{t,τ}, τ)
```

#### Flow-matching training target

For each training batch element, the code samples:

```text
A_{t,0} ~ N(0, I)
τ ~ Uniform(0, 1)
```

where `A_{t,0}` has the same shape as the clean action chunk `A_t`.

Then it builds the interpolation:

```text
A_{t,τ} = τ A_t + (1-τ) A_{t,0}
```

and uses the straight-line target velocity:

```text
A_t - A_{t,0}
```

So the flow policy trains with:

```text
v_θ(o_t, A_{t,τ}, τ) ≈ A_t - A_{t,0}
```

and the implemented loss is again an MSE, but now between:

- predicted velocity tensor
- target velocity tensor

Important details:

- for one fixed pair `(A_t, A_{t,0})`, the target velocity tensor is constant across all `τ`
- if you keep the same clean chunk `A_t` but resample a different noise tensor, the target changes
- the target is a full chunk-shaped tensor, not one scalar for the whole chunk

So different action positions inside the same chunk can have different velocity vectors.

## 4. Training Loop

The main logic lives in `hw1/src/hw1_imitation/train.py`.

### 4.1 One training step

In this code, one **training step** means:

- one batch from the DataLoader
- one forward pass
- one backward pass
- one optimizer update

This happens inside:

```python
for state, action_chunk in loader:
```

### 4.2 Batch device transfer

The code does:

```python
state = state.to(device, non_blocking=True)
action_chunk = action_chunk.to(device, non_blocking=True)
```

`non_blocking=True` is a performance hint. If the source tensors are in pinned CPU memory and the destination is CUDA, PyTorch may be able to issue the copy asynchronously.

It does not change correctness. On CPU, it has little effect.

### 4.3 The optimizer update

The `train_step` function does:

1. `optimizer.zero_grad(set_to_none=True)`
2. `loss = model.compute_loss(...)`
3. `loss.backward()`
4. `optimizer.step()`

Then it returns:

```python
loss.detach()
```

Why detach?

- we only need the loss value for logging
- we do not want to keep autograd history attached to the returned tensor

`detach()` does **not** move the tensor to CPU. It only removes gradient tracking.

### 4.4 Why `torch.compile` is used

The code attempts to compile the train step for speed:

```python
compiled_train_step = torch.compile(train_step)
```

On some machines this speeds training up significantly. On this workspace, runtime compilation can fail because the path contains a space. The loop therefore falls back to eager execution if the compiled version fails.

### 4.5 Logging intervals

The training loop tracks:

- `global_step`: total number of batches processed across all epochs
- `running_loss`: accumulated batch losses since the last log
- `interval_steps`: number of batches since the last log
- `interval_start_time`: timer for throughput measurement

When:

```python
global_step % config.log_interval == 0
```

the code logs:

```python
"train/loss": running_loss / interval_steps
"train/steps_per_sec": interval_steps / elapsed
"train/epoch": epoch + 1
```

#### Meaning of `train/loss`

This is the average batch loss over the most recent logging interval. It is not just the last single batch loss.

#### Meaning of `train/steps_per_sec`

This is the number of training steps (batches) processed per second over the recent interval.

If 100 batches took 5 seconds:

```text
steps_per_sec = 100 / 5 = 20
```

#### Why `max(elapsed, 1e-8)`?

The code uses:

```python
interval_steps / max(elapsed, 1e-8)
```

This is just a guard against division by zero or a numerically tiny denominator.

#### Why `time.perf_counter()` instead of `time.time()`?

`time.perf_counter()` is better for measuring durations:

- higher resolution
- monotonic
- not intended as wall-clock time

`time.time()` is better for timestamps, not benchmarking.

### 4.6 `model.train()` and `model.eval()`

`model.train()` puts the model in training mode.

`model.eval()` puts it in evaluation mode.

This matters for layers like:

- Dropout
- BatchNorm

Our current MLP only uses Linear/ReLU, so the difference is small here, but calling the correct mode is still standard practice.

The evaluation function calls `model.eval()`, so after eval the training loop switches back with:

```python
model.train()
```

## 5. Evaluation Rollout

The evaluation logic lives in `hw1/src/hw1_imitation/evaluation.py`.

### 5.1 What `sample_actions()` returns

For one normalized state, the policy returns one chunk of actions:

```text
(1, chunk_size, action_dim)
```

The eval code then removes the batch dimension:

```python
pred_chunk = (
    model.sample_actions(state.unsqueeze(0), num_steps=flow_num_steps)
    .cpu()
    .numpy()[0]
)
```

After `.numpy()[0]`, the shape is:

```text
(chunk_size, action_dim)
```

How that chunk is produced depends on the policy:

- MSE policy: one forward pass predicts the whole chunk directly
- flow policy: sample one initial noise chunk, denoise it for `flow_num_steps` Euler steps, then return the final chunk

### 5.2 Why `state.unsqueeze(0)`?

`state` is originally a single vector of shape `(state_dim,)`.

The model expects a batch dimension, so:

```python
state.unsqueeze(0)
```

makes it shape `(1, state_dim)`.

### 5.3 What happens after prediction?

The eval loop:

1. normalizes the current observation
2. predicts a chunk of normalized actions
3. denormalizes the chunk
4. clips actions to the env action bounds
5. executes actions one by one with `env.step(...)`
6. asks the policy for a new chunk when the current chunk is exhausted

So the flow is:

```text
current env state
-> normalize
-> predict chunk
-> execute action 1 -> env returns next state
-> execute action 2 -> env returns next state
-> ...
-> after chunk ends, predict again from the latest real env state
```

The policy does not predict the next state. The environment does.

For the flow policy, the denoising happens entirely inside `model.sample_actions(...)`. The environment only sees the final action chunk after denormalization and clipping.

### 5.4 Where evaluation data comes from

Evaluation does **not** use a separate `.zarr` evaluation dataset.

Instead, it comes directly from fresh simulator rollouts:

```python
env = gym.make("gym_pusht/PushT-v0", obs_type="state", render_mode="rgb_array")
obs, _ = env.reset(seed=ep_idx)
```

So:

- training data comes from the offline expert dataset
- evaluation data comes from the environment itself

The evaluation states are generated by:

1. the environment reset distribution
2. the environment transition dynamics under the policy's chosen actions

This is why evaluation is a stronger test than offline MSE on the training set: the policy must act **closed-loop** in the real simulator.

### 5.5 Why eval videos come from different starts

Each evaluation episode resets the environment with a different seed:

```python
obs, _ = env.reset(seed=ep_idx)
```

So the logged videos `eval/rollout_ep0`, `eval/rollout_ep1`, etc. come from different initial states. They are not repeated videos of the exact same starting condition.

### 5.6 Gymnasium `env.step(...)` returns

The line:

```python
obs, reward, terminated, truncated, info = env.step(action)
```

means:

- `obs`: next observation/state
- `reward`: scalar reward from that step
- `terminated`: task-specific terminal condition was reached
- `truncated`: episode ended for an external reason, such as a time limit
- `info`: extra debugging information

The rollout loop stops when:

```python
done = terminated or truncated
```

So yes, the environment itself decides when an episode should end.

### 5.7 Push-T reward and termination

In Push-T:

- reward is based on how much of the T block overlaps the goal zone
- success is achieved when coverage exceeds the environment's success threshold
- the environment sets `terminated=True` on success

So the evaluator does not invent its own terminal rule. It follows the environment's signals.

### 5.8 What eval reward means here

Evaluation reward is **not** the training loss.

These are different quantities:

- **Train loss**:
  - MSE policy: mean squared error between predicted clean expert actions and dataset action targets
  - flow policy: mean squared error between predicted velocity tensors and flow-matching target velocities
- **Eval reward**: task reward returned by the Push-T environment during rollout

So:

- low train loss means the model fits its supervised target well on dataset samples
- high eval reward means the policy actually performs the task successfully in the simulator

In the current evaluator, each step returns a reward, but the code summarizes one episode by storing:

```python
max_reward = max(max_reward, float(reward))
```

So for each episode, the evaluator records a single scalar: the **maximum reward reached during that episode**.

After all evaluation episodes, it logs:

```python
eval/mean_reward = mean(per_episode_max_rewards)
```

So `eval/mean_reward` is:

- one scalar for the whole evaluation pass
- computed by averaging one summary reward per episode

It is not a per-action loss and it is not used for gradient updates.

### 5.9 What `torch.no_grad()` is doing in evaluation

The evaluator wraps inference in:

```python
with torch.no_grad():
```

This means:

- no gradients are tracked
- less memory is used
- inference is faster

That is correct because evaluation is inference only, not training.

#### Evaluation is not used for backpropagation

Evaluation is only for:

- monitoring progress
- logging reward curves
- saving rollout videos
- saving checkpoints

It is **not** part of the optimization objective.

The only place that updates model parameters is the training step:

```python
loss = model.compute_loss(state, action_chunk)
loss.backward()
optimizer.step()
```

The evaluator runs under `torch.no_grad()`, so:

- no gradient graph is built
- no backward pass is run
- no optimizer step is run

So eval is purely a measurement signal, not a learning signal, in this homework.

## 6. Logging and Artifacts

### 6.1 Logger structure

The provided `Logger` writes:

1. a local CSV file for grading and plotting
2. WandB logs through `wandb.log(...)`

Training rows and evaluation rows do not have the same keys, so the code pre-creates a stable CSV header:

```python
logger.header = [
    "train/loss",
    "train/steps_per_sec",
    "train/epoch",
    "eval/mean_reward",
    "step",
]
```

This ensures both training metrics and evaluation reward appear in `log.csv`.

Example:

```csv
train/loss,train/steps_per_sec,train/epoch,eval/mean_reward,step
0.50,13.0,1,,47
,,,0.10,47
```

Blank fields simply mean that row did not log that metric.

### 6.2 WandB behavior

WandB is used for:

- scalar metrics
- videos
- checkpoints / artifacts

#### During training

`wandb.init(...)` starts a run.

`logger.log(...)` sends:

- training metrics
- evaluation metrics
- evaluation videos

#### During evaluation

The code also saves a model checkpoint and logs it as an artifact.

#### Local vs remote

If WandB is online and you are logged in:

- data is saved locally
- data is also synced to the WandB cloud dashboard

If WandB is run in offline mode:

- data is saved locally only
- later you can sync it manually with `wandb sync ...`

#### If you do not have a WandB account

You can still test locally with:

```bash
WANDB_MODE=offline uv run python -m hw1_imitation.train ...
```

But for the homework's required WandB logs/videos, using a real WandB account is the safer route.

### 6.3 `logger.dump_for_grading()`

At the end of training, the code calls:

```python
logger.dump_for_grading()
```

This:

1. finishes the WandB run
2. copies the WandB run directory into the experiment folder

So the experiment directory contains:

- `log.csv`
- copied WandB files
- checkpoints
- rollout videos if evaluation logged them

This is important for grading and for local inspection after the run finishes.

## 7. Summary and Overfitting

The core mental model is:

### 7.1 Training

```text
expert dataset
-> normalize state/action
-> supervised learning
-> if MSE policy: predict clean action chunks
-> if flow policy: predict velocities on noisy chunks
-> optimize the corresponding supervised loss
```

### 7.2 Evaluation

```text
real env state
-> normalize
-> produce an action chunk
-> denormalize
-> execute actions in Gym one by one
-> Gym returns next state/reward/done
-> repeat
```

The policy predicts actions. The environment produces next states and determines reward/termination.

For the flow policy, the final predicted actions come from denoising a sampled noise chunk with Euler integration before execution.

### 7.3 About train/eval overlap and overfitting

Since evaluation comes from the same Push-T environment family, it is not a completely different task. There may be conceptual overlap between:

- the kinds of states seen in the expert dataset
- the kinds of states reached during evaluation rollouts

But evaluation is still not the same as checking performance on training rows.

During rollout:

- the policy's own actions affect future states
- mistakes can compound over time
- the policy may drift into states not well covered by the expert demonstrations

That is why a policy can have:

- low training loss
- but weak evaluation reward

This is the standard imitation learning gap between:

- **offline label matching**
- **closed-loop control performance**

So yes, overfitting is possible. A common sign is:

- training loss keeps improving
- but evaluation reward plateaus or gets worse

That is why the homework asks you to run and log `evaluate_policy(...)` periodically instead of relying only on the training loss.
