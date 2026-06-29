"""Fast norm-stats computation for LeRobot parquet datasets.

This script avoids loading video frames and computes stats directly from parquet
columns (state/action), which is much faster for large multi-camera datasets.

Designed for configs like pi05_indust_ee where:
- data source is LeRobot local dataset (repo_id is a local path)
- state comes from "observation.state"
- action comes from "action"
- optional extra delta transform is applied on first 6 action dims
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import tqdm
import tyro

import openpi.shared.normalize as normalize
import openpi.training.config as _config


def _to_2d_array(col_np: np.ndarray) -> np.ndarray:
    """Convert parquet column to shape [N, D]."""
    arr = np.asarray(col_np)
    if arr.dtype == object:
        arr = np.stack(arr.tolist(), axis=0)
    if arr.ndim == 1:
        arr = arr[:, None]
    return arr


def _make_action_chunks(actions: np.ndarray, horizon: int) -> np.ndarray:
    """Create [N, H, D] action chunks by tail-padding with last action."""
    if horizon <= 1:
        return actions[:, None, :]

    n, d = actions.shape
    pad = np.repeat(actions[-1:, :], horizon - 1, axis=0)
    ap = np.concatenate([actions, pad], axis=0)

    s0, s1 = ap.strides
    chunks = np.lib.stride_tricks.as_strided(
        ap,
        shape=(n, horizon, d),
        strides=(s0, s0, s1),
        writeable=False,
    )
    return np.asarray(chunks)


def main(
    config_name: str,
    max_frames: int | None = None,
    state_key: str = "observation.state",
    action_key: str = "action",
    repo_id: str | None = None,
):
    """Compute fast norm stats from LeRobot parquet columns.

    Args:
        repo_id: Optional override of the dataset path so a single config (e.g.
            ``pine_foundry``) can be retargeted at each per-dataset directory and
            at the merged directory. Stats are written to ``assets_dirs / repo_id``
            (= the dataset dir when repo_id is an absolute path), which is exactly
            where training loads them from.
    """
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)

    if data_config.rlds_data_dir is not None:
        raise ValueError("This fast script currently supports LeRobot parquet datasets, not RLDS.")

    repo_id_eff = repo_id if repo_id is not None else data_config.repo_id
    if repo_id_eff is None:
        raise ValueError("Data config must have a repo_id (or pass --repo-id)")

    ds_root = Path(repo_id_eff)
    if not ds_root.exists():
        raise FileNotFoundError(f"Dataset path not found: {ds_root}")

    data_dir = ds_root / "data"
    parquet_files = sorted(data_dir.glob("chunk-*/episode_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under: {data_dir}")

    # Whether to mirror the extra delta transform used in pi05_indust_ee config.
    extra_delta_transform = bool(getattr(config.data, "extra_delta_transform", False))
    action_horizon = int(config.model.action_horizon)

    stats = {
        "state": normalize.RunningStats(),
        "actions": normalize.RunningStats(),
    }

    # For user-friendly progress, prefer total_frames from meta/info.json if available.
    total_frames = None
    info_path = ds_root / "meta" / "info.json"
    if info_path.exists():
        try:
            total_frames = int(json.loads(info_path.read_text()).get("total_frames"))
        except Exception:
            total_frames = None

    frames_budget = max_frames if max_frames is not None else total_frames
    processed_frames = 0

    pbar = tqdm.tqdm(parquet_files, desc="Computing stats (fast parquet mode)")
    for pq_path in pbar:
        table = pq.read_table(pq_path, columns=[state_key, action_key])

        state = _to_2d_array(table[state_key].to_numpy()).astype(np.float32)
        action = _to_2d_array(table[action_key].to_numpy()).astype(np.float32)

        if state.shape[0] != action.shape[0]:
            raise ValueError(f"Row mismatch in {pq_path}: state={state.shape[0]}, action={action.shape[0]}")

        if max_frames is not None:
            remain = max_frames - processed_frames
            if remain <= 0:
                break
            if state.shape[0] > remain:
                state = state[:remain]
                action = action[:remain]

        # state stats are on [N, D_state]
        stats["state"].update(state)

        # actions stats are on action chunks [N, H, D_action]
        action_chunks = _make_action_chunks(action, action_horizon)

        # Mirror extra delta transform: first 6 action dims minus current state first 6 dims.
        if extra_delta_transform:
            dims = min(6, action_chunks.shape[-1], state.shape[-1])
            action_chunks = action_chunks.copy()  # make writable
            action_chunks[..., :dims] -= state[:, None, :dims]

        stats["actions"].update(action_chunks)

        processed_frames += state.shape[0]
        pbar.set_postfix({"frames": processed_frames})

    if processed_frames < 2:
        raise ValueError(f"Not enough frames to compute stats: {processed_frames}")

    # Pad raw-dim stats (state=D_state, actions=D_action) up to the model action_dim,
    # mirroring scripts/compute_norm_stats.py which computes on the policy Inputs'
    # zero-padded [*, action_dim] tensors. openpi's Normalize slices stats to the data
    # last-dim, so stats shorter than action_dim break training; padded dims get
    # all-zero stats (== stats of the zero-pad), so they normalize consistently.
    action_dim = int(config.model.action_dim)

    def _pad(arr):
        if arr is None:
            return None
        arr = np.asarray(arr)
        if arr.ndim == 1 and arr.shape[0] < action_dim:
            arr = np.concatenate([arr, np.zeros(action_dim - arr.shape[0], dtype=arr.dtype)])
        return arr

    def _pad_stats(ns):
        return normalize.NormStats(mean=_pad(ns.mean), std=_pad(ns.std), q01=_pad(ns.q01), q99=_pad(ns.q99))

    norm_stats = {k: _pad_stats(v.get_statistics()) for k, v in stats.items()}

    output_path = config.assets_dirs / repo_id_eff
    print(f"Processed frames: {processed_frames}")
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
