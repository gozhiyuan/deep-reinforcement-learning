"""Dataset utilities for Push-T."""

from __future__ import annotations

import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset

PUSHT_URL = "https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip"
ZARR_RELATIVE_PATH = Path("pusht") / "pusht_cchi_v7_replay.zarr"


@dataclass(frozen=True)
class Normalizer:
    """Feature-wise normalizer for states and actions."""

    state_mean: np.ndarray
    state_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray

    @staticmethod
    def _safe_std(std: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        return np.maximum(std, eps)

    @classmethod
    def from_data(cls, states: np.ndarray, actions: np.ndarray) -> "Normalizer":
        state_mean = states.mean(axis=0)
        state_std = cls._safe_std(states.std(axis=0))
        action_mean = actions.mean(axis=0)
        action_std = cls._safe_std(actions.std(axis=0))
        return cls(state_mean, state_std, action_mean, action_std)

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        return (state - self.state_mean) / self.state_std

    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        return (action - self.action_mean) / self.action_std

    def denormalize_action(self, action: np.ndarray) -> np.ndarray:
        return action * self.action_std + self.action_mean


def download_pusht(dataset_dir: Path) -> Path:
    """Download and extract the Push-T dataset if needed.

    Returns the path to the extracted Zarr dataset.
    """

    dataset_dir.mkdir(parents=True, exist_ok=True)
    zarr_path = dataset_dir / ZARR_RELATIVE_PATH
    if zarr_path.exists():
        return zarr_path

    zip_path = dataset_dir / "pusht.zip"
    if not zip_path.exists():
        urllib.request.urlretrieve(PUSHT_URL, zip_path)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(dataset_dir)

    return zarr_path


def load_pusht_zarr(zarr_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    root = zarr.open(zarr_path, mode="r")
    states = np.asarray(root["data"]["state"][:], dtype=np.float32)
    actions = np.asarray(root["data"]["action"][:], dtype=np.float32)
    episode_ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
    return states, actions, episode_ends


def build_valid_indices(episode_ends: np.ndarray, chunk_size: int) -> np.ndarray:
    """Build valid starting indices for sampling chunks of data within episodes.

    This function generates indices where each index is a valid starting point
    for extracting a chunk of 'chunk_size' consecutive elements (e.g., actions)
    from the dataset, ensuring chunks don't span multiple episodes or go out of bounds.

    Args:
        episode_ends: Array of ending indices (exclusive) for each episode.
        chunk_size: The length of each chunk to sample.

    Returns:
        Array of valid starting indices for chunks.

    Examples:
        Suppose episode_ends = np.array([50, 100, 150]) and chunk_size = 20:
        - Episodes span indices: 0-49, 50-99, 100-149.
        - For episode 1 (0-49): last_start = 50 - 20 = 30 >= 0, so add indices 0 to 30.
        - For episode 2 (50-99): last_start = 100 - 20 = 80 >= 50, so add 50 to 80.
        - For episode 3 (100-149): last_start = 150 - 20 = 130 >= 100, so add 100 to 130.
        - Result: np.array([0, 1, ..., 30, 50, 51, ..., 80, 100, ..., 130])

        If an episode is too short (e.g., episode_ends = [10, 100], chunk_size=20):
        - Episode 1 (0-9): last_start = 10 - 20 = -10 < 0, so skip.
        - Episode 2 (10-99): last_start = 100 - 20 = 80 >= 10, so add 10 to 80.
        - Result: np.array([10, 11, ..., 80])
    """
    # Compute the starting index for each episode (first episode starts at 0,
    # subsequent episodes start right after the previous episode's end)
    starts = np.concatenate(([0], episode_ends[:-1]))
    indices: list[int] = []
    # Iterate over each episode's start and end indices
    for start, end in zip(starts, episode_ends, strict=True):
        # Calculate the latest possible start index for a chunk within this episode
        last_start = end - chunk_size
        # Skip episodes that are too short to contain a full chunk
        if last_start < start:
            continue
        # Add all valid starting indices for this episode (from start to last_start inclusive)
        indices.extend(range(start, last_start + 1))
    return np.asarray(indices, dtype=np.int64)


class PushtChunkDataset(Dataset):
    """Dataset of (state, action_chunk) pairs using a sliding window.

    This dataset creates samples for imitation learning by pairing each state
    with a future chunk of actions. It uses valid starting indices (computed via
    build_valid_indices) to ensure chunks stay within episode boundaries.

    Each sample is a tuple (state, action_chunk), where:
    - state: The state at time t (a single vector).
    - action_chunk: A sequence of chunk_size consecutive actions starting from t.

    This design enables the model to predict multi-step action sequences from the current state.

    Args:
        states: Array of all states in the dataset (shape: [total_steps, state_dim]).
        actions: Array of all actions in the dataset (shape: [total_steps, action_dim]).
        episode_ends: Array of ending indices for each episode.
        chunk_size: Number of consecutive actions in each chunk.
        normalizer: Optional normalizer for states and actions.

    Examples:
        Suppose states/actions have 1000 total steps, episode_ends=[100, 200, 1000], chunk_size=5:
        - Valid indices might be [0, 1, ..., 95, 100, ..., 195, 200, ..., 995].
        - For idx=0: t=0, returns (states[0], actions[0:5])
        - For idx=95: t=95, returns (states[95], actions[95:100])
        - For idx=96: t=100, returns (states[100], actions[100:105])  # Next episode
    """

    def __init__(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        episode_ends: np.ndarray,
        chunk_size: int,
        normalizer: Normalizer | None = None,
    ) -> None:
        # Store the raw states and actions arrays
        self.states = states
        self.actions = actions
        # Store chunk size for slicing actions
        self.chunk_size = chunk_size
        # Optional normalizer for data preprocessing
        self.normalizer = normalizer
        # Compute valid starting indices for chunks (ensures no cross-episode or out-of-bounds samples)
        self.indices = build_valid_indices(episode_ends, chunk_size)

    def __len__(self) -> int:
        # Return the total number of valid samples (one per valid starting index)
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Get the starting time index for this sample
        t = int(self.indices[idx])
        # Extract the state at time t
        state = self.states[t]
        # Extract the chunk of actions starting from t (length: chunk_size)
        action_chunk = self.actions[t : t + self.chunk_size]

        # Apply normalization if a normalizer is provided
        if self.normalizer is not None:
            state = self.normalizer.normalize_state(state)
            action_chunk = self.normalizer.normalize_action(action_chunk)

        # Return as PyTorch tensors (state as float, action_chunk as float)
        return (
            torch.from_numpy(state).float(),
            torch.from_numpy(action_chunk).float(),
        )
