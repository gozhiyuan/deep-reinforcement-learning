# Flow Matching for Action Chunks

This note explains the flow-matching policy described in class notes, with the notation cleaned up and the examples organized around the main questions:

- what the variables mean
- why the target is `A_t - A_{t,0}`
- what is fixed in training and what changes in inference
- what shapes the tensors have
- why this helps with multimodal action chunks

## 1. Overview

An MSE policy predicts the whole action chunk in one shot. That can work when there is basically one correct future, but it struggles when the same observation admits multiple valid action chunks.

Example:

- one valid chunk goes left around an obstacle
- another valid chunk goes right around it

A direct MSE model often predicts something near the average, which may be invalid.

Flow matching changes the problem. Instead of directly predicting the final chunk, it learns a **velocity field** that transforms noise into a realistic action chunk.

So instead of asking:

> "What is the final chunk?"

it asks:

> "From the current noisy point, which direction should I move next?"

## 2. Notation and Shapes

### 2.1 Two Different Time Variables

There are two different time variables:

| Symbol | Meaning |
|---|---|
| `t` | environment / dataset timestep |
| `τ` | flow time used for interpolation and denoising |

So:

- `t` tells you which observation-action pair from the dataset you are looking at
- `τ` tells you where you are on the path from noise to data

Important notation:

- `A_t` = clean action chunk at environment time `t`
- `A_{t,0}` = noise chunk at flow time `τ = 0`
- `A_{t,τ}` = interpolated chunk at flow time `τ`

The `0` in `A_{t,0}` means flow time `0`, not environment time `0`.

### 2.2 Action Chunk Shape

If the chunk length is `H` and each action has dimension `d_a`, then:

```text
A_t ∈ R^{H x d_a}
```

With batch size `B`:

```text
A_t ∈ R^{B x H x d_a}
```

Conceptually this means:

- one chunk contains `H` future actions
- each action is a `d_a`-dimensional vector

Example with `H = 3`, `d_a = 2`:

```text
A_t =
[
  [ 0.40,  0.10],
  [ 0.50,  0.00],
  [ 0.60, -0.10]
]
```

### 2.3 Observation `o_t`

`o_t` is the observation available at environment time `t`.

Its shape depends on the environment.

Common cases:

1. Low-dimensional state:

   ```text
   o_t ∈ R^{d_o}
   ```

   Example:

   ```text
   o_t = [0.12, -0.30, 0.08, 0.55]
   ```

2. Image observation:

   ```text
   o_t ∈ R^{C x H x W}
   ```

   Example:

   ```text
   o_t ∈ R^{3 x 84 x 84}
   ```

3. Robot observation with multiple parts:

   ```text
   o_t = {
     image: [3, 128, 128],
     state: [14]
   }
   ```

Concrete vector-state example:

```text
o_t = [1.2, -0.4, 0.8, 0.3, -0.1, 2.5]
```

If `A_{t,τ}` has shape `[3, 2]`, then one conceptual network input is:

```text
(
  o_t,        shape [6]
  A_{t,τ},    shape [3, 2]
  τ           shape [1]
)
```

### 2.4 Practical Architecture Note

When we write

```text
v_θ(o_t, A_{t,τ}, τ),
```

that means the network is conditioned on all three inputs.

In practice, the model usually does **not** flatten raw image pixels, the action chunk tensor, and the scalar `τ` into one long vector.

More commonly it does:

1. encode `o_t`
2. encode `A_{t,τ}`
3. embed `τ`
4. fuse those representations
5. output a tensor with the same shape as the action chunk

So the output shape is:

```text
v_θ(o_t, A_{t,τ}, τ) ∈ R^{H x d_a}
```

or with batch:

```text
v_θ(o_t, A_{t,τ}, τ) ∈ R^{B x H x d_a}
```

## 3. Training Path and Target Velocity

### 3.1 Interpolation Path

For a training sample `(o_t, A_t)`, flow matching samples:

```text
A_{t,0} ~ N(0, I)
τ ~ Uniform(0,1)
```

and defines the interpolation:

```text
A_{t,τ} = τ A_t + (1-τ) A_{t,0}
```

This is a straight line from noise to the clean action chunk:

```text
τ=0                                      τ=1
A_{t,0} -------------------------------- A_t
 noise             A_{t,τ}                data
```

At the endpoints:

```text
τ = 0  =>  A_{t,τ} = A_{t,0}
τ = 1  =>  A_{t,τ} = A_t
```

### 3.2 What Does `τ(10)` Mean?

It is ordinary scalar multiplication.

If `A_t = 10`, then:

```text
τ(10) = 10τ
```

So if:

```text
A_{t,τ} = τ(10) + (1-τ)(2)
```

then:

```text
A_{t,τ} = 10τ + 2 - 2τ = 2 + 8τ
```

That means:

- start at `2`
- move a total distance of `8`
- `τ` tells you what fraction of that distance you have moved

### 3.3 Why the Target Is `A_t - A_{t,0}`

Differentiate the interpolation with respect to flow time `τ`:

```text
A_{t,τ} = τ A_t + (1-τ) A_{t,0}
```

so

```text
dA_{t,τ}/dτ = A_t - A_{t,0}
```

This is the velocity of the straight path from noise to data.

That is why the training target is:

```text
A_t - A_{t,0}
```

and the loss is:

```text
L_FM(θ) = (1/B) Σ || v_θ(o_t, A_{t,τ}, τ) - (A_t - A_{t,0}) ||_2^2
```

### 3.4 What Is a "Path"?

A path is defined by a fixed pair:

- clean chunk `A_t`
- sampled noise chunk `A_{t,0}`

Once those are fixed, the straight-line path is fixed, and so is its target velocity tensor:

```text
A_t - A_{t,0}
```

Important consequences:

- same `A_t` and same `A_{t,0}` -> same target velocity tensor for all `τ`
- same `A_t` but different sampled `A_{t,0}` -> different path, usually different target velocity tensor

So training does **not** use only one path per clean chunk. The same clean chunk can appear with many different sampled noise tensors:

```text
(o_t, A_t, A_{t,0}^{(1)}, τ^{(1)})
(o_t, A_t, A_{t,0}^{(2)}, τ^{(2)})
(o_t, A_t, A_{t,0}^{(3)}, τ^{(3)})
...
```

All the noise tensors above are independent samples from the same Gaussian `N(0, I)`.

### 3.5 Scalar Example

Take:

```text
A_t = 10
A_{t,0} = 2
```

Then:

```text
A_{t,τ} = τ(10) + (1-τ)(2) = 2 + 8τ
```

and:

```text
dA_{t,τ}/dτ = 8 = 10 - 2
```

So for this fixed path:

- `τ = 0.25` -> input point `A_{t,τ} = 4`, target velocity `8`
- `τ = 0.50` -> input point `A_{t,τ} = 6`, target velocity `8`
- `τ = 0.75` -> input point `A_{t,τ} = 8`, target velocity `8`

The point changes with `τ`, but the target velocity stays constant for this path.

If we keep the same clean chunk but change the noise, the target changes:

```text
A_t = 10, A_{t,0} = 2   => target velocity = 8
A_t = 10, A_{t,0} = -3  => target velocity = 13
A_t = 10, A_{t,0} = 7   => target velocity = 3
```

So the constant target belongs to one fixed `(A_t, A_{t,0})` path, not to the clean chunk alone.

### 3.6 Tensor Example With Chunk Dimensions

Suppose:

- chunk length `H = 3`
- action dimension `d_a = 2`

Let:

```text
A_t =
[
  [ 0.40,  0.10],
  [ 0.50,  0.00],
  [ 0.60, -0.10]
]
```

and:

```text
A_{t,0} =
[
  [-0.20,  0.30],
  [ 0.10, -0.40],
  [ 0.00,  0.20]
]
```

Sample:

```text
τ = 0.25
```

Then:

```text
A_{t,τ} = 0.25 A_t + 0.75 A_{t,0}
```

which gives:

```text
A_{t,τ} =
[
  [-0.05,  0.250],
  [ 0.20, -0.300],
  [ 0.15,  0.125]
]
```

The target velocity is:

```text
A_t - A_{t,0} =
[
  [ 0.60, -0.20],
  [ 0.40,  0.40],
  [ 0.60, -0.30]
]
```

Two important clarifications:

1. This target tensor is fixed across all `τ` for this path.
2. The entries inside the tensor are **not** all the same.

That second point matters: the model is **not** learning one scalar speed for the whole chunk. It is learning a chunk-shaped velocity tensor. Different positions inside the chunk can have different velocity vectors:

- first action in the chunk: `[0.60, -0.20]`
- second action in the chunk: `[0.40, 0.40]`
- third action in the chunk: `[0.60, -0.30]`

## 4. Training Procedure

For each minibatch example:

1. take a real training pair `(o_t, A_t)`
2. sample noise `A_{t,0} ~ N(0, I)`
3. sample `τ ~ U(0,1)`
4. compute:

   ```text
   A_{t,τ} = τ A_t + (1-τ) A_{t,0}
   ```

5. train the model so that:

   ```text
   v_θ(o_t, A_{t,τ}, τ) ≈ A_t - A_{t,0}
   ```

So training teaches:

> "Given an observation, an intermediate noisy chunk, and a flow time, predict the velocity that moves along the chosen path toward the clean chunk."

## 5. Inference and Euler Integration

### 5.1 What Is the ODE?

ODE means **ordinary differential equation**.

At inference, the true clean chunk `A_t` is unknown, so we cannot compute `A_t - A_{t,0}` directly.

Instead we start from fresh noise:

```text
A_{t,0} ~ N(0, I)
```

and integrate:

```text
dA_{t,τ}/dτ = v_θ(o_t, A_{t,τ}, τ),   from τ=0 to τ=1
```

This means:

> "As flow time `τ` changes a little, how should the current sample change?"

### 5.2 Euler Integration

The simplest numerical solver is Euler integration:

```text
A_{t,τ+1/n} = A_{t,τ} + (1/n) v_θ(o_t, A_{t,τ}, τ)
```

where `n` is the number of denoising steps.

Interpretation:

- divide `[0,1]` into `n` equal pieces
- at each step, ask the model for a velocity
- move a small amount in that direction

`n` can be any positive integer.

- larger `n` -> smaller step size `Δτ = 1/n`, more granular integration, usually better approximation, slower
- smaller `n` -> larger step size, faster, rougher approximation

In hand-worked examples, `n = 4` is often used only because the arithmetic is simple.

### 5.3 One Noise Sample Per Inference Run

For one inference run, you usually:

1. sample one initial noise tensor
2. denoise that one sample from `τ = 0` to `τ = 1`
3. obtain one final action chunk `A_{t,1}`

If you want multiple candidate chunks for the same observation, you run inference multiple times with different initial noise samples.

### 5.4 The Whole Chunk Is Denoised Jointly

Euler integration runs over **flow time** `τ`, not over the action index inside the chunk.

So the model does **not** generate the chunk autoregressively like:

1. predict first action
2. use first action to predict second
3. use second to predict third

Instead, the whole chunk is updated together at each Euler step.

If:

```text
A_{t,τ} ∈ R^{H x d_a}
```

then:

```text
v_θ(o_t, A_{t,τ}, τ) ∈ R^{H x d_a}
```

and one Euler step updates all action positions simultaneously.

### 5.5 Why the Predicted Velocity Can Vary During Inference

In training, for one fixed path `(A_t, A_{t,0})`, the target is constant across `τ`.

But in inference:

- the clean target `A_t` is unknown
- the model only sees the current sample `A_{t,τ}`
- after each Euler step, both `A_{t,τ}` and `τ` change

So the model recomputes:

```text
v_θ(o_t, A_{t,τ}, τ)
```

at every step, and the prediction can vary.

Still, if the model is good and the rollout stays near a training-style path, the predicted velocity should usually stay **close** to the training-style target, not jump wildly.

For example, if the underlying training-style target is around `6`, a good rollout may produce:

```text
6.1, 5.9, 5.8, 6.0
```

not:

```text
100, 1000
```

So the right intuition is:

- exact training label for one path: constant
- inference prediction for a good model: usually close to that constant, with small smooth variation

### 5.6 Euler Example

Take a 1D toy example with initial noise:

```text
A_{t,0} = -2
```

Suppose a good model predicts velocities close to `6` and we choose `n = 4`, so `Δτ = 0.25`.

Use:

```text
v_1 = 6.1
v_2 = 5.9
v_3 = 5.8
v_4 = 6.0
```

Then:

1. step 1:

   ```text
   A = -2 + 0.25 * 6.1 = -0.475
   ```

2. step 2:

   ```text
   A = -0.475 + 0.25 * 5.9 = 1.000
   ```

3. step 3:

   ```text
   A = 1.000 + 0.25 * 5.8 = 2.450
   ```

4. step 4:

   ```text
   A = 2.450 + 0.25 * 6.0 = 3.950
   ```

So the sample moves gradually from noise toward a clean value, while the predicted velocity stays near the learned target instead of being exactly identical at every step.

## 6. Why Not Predict the Final Chunk Directly?

You can directly predict the final chunk. That is what a standard MSE policy does.

The issue is that one-shot regression is often a bad objective when the conditional action distribution is multimodal.

Direct regression asks:

> "Given `o_t`, what single chunk should I output?"

Flow matching asks:

> "Given `o_t`, a current noisy guess, and `τ`, what local update should I make?"

Why this can help:

1. Local prediction is often easier than predicting the whole structured chunk in one shot.
2. The same clean chunk can generate many supervised training states by resampling noise and `τ`.
3. Different initial noise samples at inference can produce different valid outputs.

## 7. Why MSE Struggles With Multimodality

Suppose for the same observation there are two valid chunks:

```text
A_t^(left)  = -10
A_t^(right) = +10
```

A direct MSE prediction `a` minimizes:

```text
0.5 (a + 10)^2 + 0.5 (a - 10)^2
```

which is minimized at:

```text
a = 0
```

But `0` is not either valid mode. It is just the average.

Flow matching is different because different sampled noises can flow to different valid modes:

```text
o_t + noise z_1 -> left chunk
o_t + noise z_2 -> right chunk
```

So it models a distribution over chunks rather than one average chunk.

## 8. Relation to Diffusion

Flow matching is similar to diffusion because both:

- start from noise
- use a time-conditioned neural network
- iteratively transform noise into data

The main conceptual distinction is:

- diffusion: learn to reverse a noising process
- flow matching: learn a velocity field for a transport path

## 9. Compact Summary

### Training

For each training example:

1. sample `(o_t, A_t)` from the dataset
2. sample noise `A_{t,0} ~ N(0, I)`
3. sample `τ ~ U(0,1)`
4. build:

   ```text
   A_{t,τ} = τ A_t + (1-τ) A_{t,0}
   ```

5. train:

   ```text
   v_θ(o_t, A_{t,τ}, τ) ≈ A_t - A_{t,0}
   ```

### Inference

For one generated chunk:

1. sample one initial noise tensor `A_{t,0} ~ N(0, I)`
2. integrate:

   ```text
   dA_{t,τ}/dτ = v_θ(o_t, A_{t,τ}, τ)
   ```

3. approximate with Euler:

   ```text
   A_{t,τ+1/n} = A_{t,τ} + (1/n) v_θ(o_t, A_{t,τ}, τ)
   ```

4. use `A_{t,1}` as the final action chunk

### One-Sentence Intuition

Flow matching does not ask the policy to guess the final action chunk immediately; it asks the policy to learn how to continuously steer noise into a realistic action chunk conditioned on the current observation.
