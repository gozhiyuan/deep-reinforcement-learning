# Push-T Dataset Structure Guide

A comprehensive explanation of how the dataset, data loading, and shuffling work in the imitation learning pipeline.

---

## Overview

The Push-T imitation learning pipeline uses a custom `PushtChunkDataset` that provides efficient, lazy-loaded access to state-action pairs organized as:
- **Input**: A single state vector at time `t`
- **Output**: A sequence (chunk) of `chunk_size` consecutive actions starting from time `t`

This design enables the model to predict multi-step action sequences from the current state.

---

## 1. Data Loading: Raw Data

### What Gets Loaded

When `load_pusht_zarr()` is called, it loads all data across all episodes:

```python
states, actions, episode_ends = load_pusht_zarr(zarr_path)
```

- **`states`**: NumPy array of shape `[total_steps, state_dim]`
  - Example: `[5000, 10]` means 5000 timesteps, each with 10-dimensional state
  - Contains all states from all episodes in chronological order

- **`actions`**: NumPy array of shape `[total_steps, action_dim]`
  - Example: `[5000, 2]` means 5000 timesteps, each with 2-dimensional action
  - Contains all actions from all episodes in chronological order

- **`episode_ends`**: NumPy array of shape `[num_episodes]`
  - Example: `[100, 200, 350]` means:
    - Episode 1: indices 0 to 99 (100 steps)
    - Episode 2: indices 100 to 199 (100 steps)
    - Episode 3: indices 200 to 349 (150 steps)
  - Each value is the **exclusive** end index of an episode

### Key Point
All raw data is loaded into memory once and **never duplicated**. The dataset simply references these arrays.

---

## 2. Build Valid Indices: The Pointer Array

### Purpose

Not all timesteps can be valid starting points for action chunks. We need indices where a full chunk fits within an episode boundary.

### Function: `build_valid_indices(episode_ends, chunk_size)`

```python
def build_valid_indices(episode_ends: np.ndarray, chunk_size: int) -> np.ndarray:
    starts = np.concatenate(([0], episode_ends[:-1]))
    indices: list[int] = []
    for start, end in zip(starts, episode_ends, strict=True):
        last_start = end - chunk_size
        if last_start < start:
            continue
        indices.extend(range(start, last_start + 1))
    return np.asarray(indices, dtype=np.int64)
```

### Step-by-Step Example

Suppose `episode_ends = [50, 100, 150]` and `chunk_size = 20`:

1. **Compute Episode Starts**
   - `starts = [0, 50, 100]` (first episode at 0, others right after previous end)

2. **For Episode 1 (indices 0-49)**
   - `last_start = 50 - 20 = 30`
   - `30 >= 0`? Yes, so add indices `[0, 1, 2, ..., 30]`
   - Last chunk: `actions[30:50]` ✓ (20 actions)

3. **For Episode 2 (indices 50-99)**
   - `last_start = 100 - 20 = 80`
   - `80 >= 50`? Yes, so add indices `[50, 51, ..., 80]`
   - Last chunk: `actions[80:100]` ✓ (20 actions)

4. **For Episode 3 (indices 100-149)**
   - `last_start = 150 - 20 = 130`
   - `130 >= 100`? Yes, so add indices `[100, 101, ..., 130]`
   - Last chunk: `actions[130:150]` ✓ (20 actions)

5. **Result**
   ```
   self.indices = [0, 1, 2, ..., 30, 50, 51, ..., 80, 100, 101, ..., 130]
   ```

### Edge Case: Short Episodes

If `episode_ends = [10, 100]` and `chunk_size = 20`:
- **Episode 1 (indices 0-9)**: `last_start = 10 - 20 = -10 < 0` → **Skip entirely**
- **Episode 2 (indices 10-99)**: `last_start = 100 - 20 = 80 >= 10` → Add `[10, 11, ..., 80]`

Episodes shorter than `chunk_size` are excluded from the dataset.

### Key Insight

`self.indices` is a **pointer array** that acts like a database index. Each value points to a valid starting timestep in the raw `states` and `actions` arrays.

---

## 3. PushtChunkDataset: Lazy Loading

### Structure

The dataset stores:

```python
class PushtChunkDataset(Dataset):
    def __init__(self, states, actions, episode_ends, chunk_size, normalizer=None):
        self.states = states              # Full raw states array (reference, not copy)
        self.actions = actions            # Full raw actions array (reference, not copy)
        self.chunk_size = chunk_size      # Integer: size of action chunks
        self.normalizer = normalizer      # Optional: normalizer object
        self.indices = build_valid_indices(episode_ends, chunk_size)  # Pointer array
```

### Lazy Loading: How Samples Are Generated

The `__getitem__` method **does not** return precomputed samples. Instead, it computes them on-the-fly:

```python
def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    # Step 1: Get the actual timestep from the pointer array
    t = int(self.indices[idx])  # e.g., idx=42 → t=523 (some valid starting index)
    
    # Step 2: Slice the raw arrays dynamically
    state = self.states[t]      # Single state vector at time t
    action_chunk = self.actions[t : t + self.chunk_size]  # chunk_size consecutive actions
    
    # Step 3: Normalize if needed
    if self.normalizer is not None:
        state = self.normalizer.normalize_state(state)
        action_chunk = self.normalizer.normalize_action(action_chunk)
    
    # Step 4: Convert to PyTorch tensors and return
    return (
        torch.from_numpy(state).float(),
        torch.from_numpy(action_chunk).float(),
    )
```

### Example Flow

Suppose the dataset has 100 valid samples and we request `dataset[42]`:

1. `__getitem__(42)` is called
2. Lookup: `t = self.indices[42]` → e.g., `t = 523`
3. Slice: `state = states[523]`, `actions = actions[523:531]` (if `chunk_size=8`)
4. Return: `(state_tensor, action_chunk_tensor)`

**No precomputation, no duplication—pure lazy loading.**

### Dataset Length

```python
def __len__(self) -> int:
    return len(self.indices)  # Number of valid starting positions
```

This is **not** the total number of timesteps, but the number of valid chunk samples.

Example: If dataset has 5000 timesteps across 3 episodes, `len(dataset)` might be 4950 (after removing invalid boundaries and short episodes).

---

## 4. DataLoader and Shuffling

### DataLoader Creation

```python
loader = DataLoader(
    dataset,
    batch_size=128,
    shuffle=True,      # Randomize access order
    drop_last=True,    # Discard incomplete final batch
)
```

### How Shuffling Works

1. **DataLoader creates an index array**
   ```
   [0, 1, 2, 3, ..., 4949]  (length = len(dataset))
   ```

2. **When `shuffle=True`, this array is shuffled**
   ```
   [2142, 5, 89, 3421, ..., 0, 842, ...]  (randomized order)
   ```

3. **For each item, DataLoader calls `dataset[shuffled_idx]`**
   - `dataset[2142]` → `__getitem__(2142)` → lookup `self.indices[2142]` → get valid `t` → slice data
   - `dataset[5]` → next item
   - `dataset[89]` → next item
   - etc.

### Without Shuffling (Hypothetical)

If `shuffle=False`, the DataLoader iterates in order `[0, 1, 2, ...]`, which means:
- First, all samples from episode 1 (e.g., indices 0 to 30)
- Then, all samples from episode 2 (e.g., indices 50 to 80)
- Then, all samples from episode 3 (e.g., indices 100 to 130)

This would be **sequential by episode**, as you described.

### Reshuffling Per Epoch

**Yes, the DataLoader reshuffles every epoch.**
- At the start of epoch 1: shuffle to `[2142, 5, 89, ...]`
- At the start of epoch 2: shuffle to `[841, 12, 3500, ...]` (different random order)
- At the start of epoch 3: shuffle to `[99, 4001, 1234, ...]` (different again)

This ensures diverse batches and prevents overfitting to episode order.

---

## 5. Data Flow Visualization

```
Raw Data (Zarr)
│
├─ states [5000, 10]
├─ actions [5000, 2]
└─ episode_ends [100, 200, 350]
│
↓ Load into Memory
│
Raw Arrays in Memory
├─ self.states [5000, 10]
├─ self.actions [5000, 2]
│
↓ Build Valid Indices
│
self.indices [0, 1, ..., 30, 50, 51, ..., 80, 100, ..., 130]
             └─ Pointer array (no data duplication)
│
↓ Create Dataset
│
PushtChunkDataset
│ (stores references to raw arrays + pointer array)
│
↓ Create DataLoader with shuffle=True
│
Shuffled Index Array [2142, 5, 89, 3421, ...]
│
↓ For Each Batch
│
For each shuffled_idx:
  t = self.indices[shuffled_idx]
  state = self.states[t]
  action_chunk = self.actions[t : t + chunk_size]
  → (state_tensor, action_chunk_tensor)
```

---

## 6. Memory Considerations

### Current Memory Usage

- `PushtChunkDataset` stores **references** (pointers) to the raw arrays, not copies.
- Expected memory: Same as loading the raw data (~size of Zarr file).
- For Push-T: ~100MB (easily fits on modern hardware).

### Large Dataset Concerns

If your dataset is **extremely large** (e.g., 100GB+):

**Option 1: Memory-Mapped Zarr**
```python
root = zarr.open(zarr_path, mode="r")
self.states = root["data"]["state"]   # Memory-mapped, not loaded
self.actions = root["data"]["action"]  # Memory-mapped, not loaded
```
Data stays on disk; only accessed chunks are loaded to RAM.

**Option 2: Stream Data in Batches**
- Create an episode-based DataLoader that loads episodes on-the-fly.

**Option 3: Downsample**
- Use only a subset of episodes or timesteps during training.

---

## 7. Why `drop_last=True`?

### Example

Dataset has 5000 samples, `batch_size=128`:
- Full batches: 128 × 39 = 4992 samples
- Remaining: 5000 - 4992 = 8 samples

**With `drop_last=True`**: 39 batches (4992 samples, last 8 discarded)
**With `drop_last=False`**: 40 batches (39 full + 1 batch of 8 samples)

### Reasons to Drop Last

1. **Batch Normalization**: Behaves differently with smaller batch sizes
2. **Gradient Stability**: Consistent batch sizes → stable gradient magnitudes
3. **GPU Memory**: Exact batch size allocation is easier
4. **Reproducibility**: More predictable training behavior

### Trade-off

- **Pros**: Cleaner, more stable training
- **Cons**: Slight data loss (8 samples per epoch)

For large datasets, this is negligible. For small datasets, use `drop_last=False` to use all data.

---

## 8. Episode Handling: All Episodes or Subsets?

### Key Principle

**The dataset uses ALL episodes**, but only valid samples from each:

- **Episode too short** (length < `chunk_size`): Entire episode skipped
- **Episode long enough**: Multiple overlapping chunk samples extracted
- **All episodes combined**: One large dataset of valid samples

### Example

Three episodes: lengths 50, 100, 30; `chunk_size=20`:

- **Episode 1** (length 50): Add samples [0, 1, ..., 30] ✓ (31 samples)
- **Episode 2** (length 100): Add samples [50, 51, ..., 80] ✓ (31 samples)
- **Episode 3** (length 30): Skip (too short) ✗

**Total dataset size**: 62 samples (all from all usable episodes)

---

## Summary

| Concept | Details |
|---------|---------|
| **Raw Data** | Full arrays of states, actions, and episode boundaries loaded into memory |
| **Pointer Array** | `self.indices` maps dataset indices to valid starting timesteps |
| **Lazy Loading** | Samples computed on-the-fly in `__getitem__`, not precomputed |
| **Shuffling** | DataLoader shuffles the pointer array order per epoch, mixing episodes |
| **All Episodes** | Dataset uses all episodes; valid samples from each combined into one dataset |
| **Memory** | Only raw data stored; pointer array is lightweight |
| **drop_last** | Discards incomplete final batch for training stability |

