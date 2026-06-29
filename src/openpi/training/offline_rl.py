"""PT4FM offline-RL on JAX openpi: AWR-weighted + CFG-conditioned flow loss + SFT anchor.

This is the JAX half of the PT4FM hybrid: the stage-4 policy update runs on openpi's
native JAX trainer (so we can resume directly from a finetuned JAX pi0.5 checkpoint and
keep JAX's stability), while stages 1-3 (returns / value / advantages) stay in the
PyTorch PT4FM/RECAP pipeline and hand off via ``meta/advantages_{tag}.parquet``.

The objective mirrors the validated PyTorch PT4FM forward
(``PT4FM/pt4fm/models/cfg_action_model.py``):

    L = L_RL + lambda_sft * L_SFT
    L_RL  = E[ w_awr(A) * ||v_theta(.|o, l_cfg) - u_t||^2 ]   over conditional samples
    L_SFT = E[ ||v_theta(.|o, l_raw) - u_t||^2 ]              over the anchor set

With ``awr.enabled=false, sft_aux.mode=reuse_unconditional, sft_aux.weight=1`` (and
``cfg.enabled=false``) the loss reduces **exactly** to ``jnp.mean(model.compute_loss)``
— i.e. plain SFT / RECAP — so this is a strict superset.

Consumed by ``scripts/train_offline_rl.py``. All knobs live in
:class:`openpi.training.config.OfflineRLConfig`.
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import openpi.models.model as _model
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as _transforms

logger = logging.getLogger("openpi.offline_rl")

_EPS = 1e-8


# --------------------------------------------------------------------------- #
# jnp method primitives
# --------------------------------------------------------------------------- #
def awr_weight(adv_w: jnp.ndarray, beta: float, w_max: float, mask: jnp.ndarray) -> jnp.ndarray:
    """Exponential advantage weight, clipped and mean-1 normalized over ``mask``.

    ``w = clip(exp((A - max_A)/beta), 0, w_max)``; the ``-max_A`` shift is for
    numerical stability and cancels after normalization. Returns 0 outside ``mask``.
    """
    a = adv_w.astype(jnp.float32)
    maskf = mask.astype(jnp.float32)
    a_ref = jnp.max(jnp.where(mask, a, -jnp.inf))
    a_ref = jnp.where(jnp.isfinite(a_ref), a_ref, 0.0)
    w = jnp.exp((a - a_ref) / beta)
    w = jnp.clip(w, 0.0, w_max) * maskf
    total = jnp.sum(w)
    n = jnp.sum(maskf)
    # normalize to mean 1 over the active set; fall back to the mask if degenerate.
    return jnp.where(total > _EPS, w * (n / (total + _EPS)), maskf)


def cfg_routing_masks(
    advantage: jnp.ndarray, positive_only_conditional: bool, uncond_prob: float, rng: jax.Array
) -> dict[str, jnp.ndarray]:
    """RECAP CFG routing (jnp port of compute_cfg_routing_masks).

    ``positive_only_conditional`` and ``uncond_prob`` are static (python) values.
    """
    advantage = advantage.astype(bool)
    b = advantage.shape[0]
    rand = jax.random.uniform(rng, (b,))
    positive = advantage
    negative = ~positive
    if positive_only_conditional:
        pos_cond = positive & (rand > uncond_prob)
        neg_cond = jnp.zeros_like(positive)
    else:
        keep = rand > uncond_prob
        pos_cond = positive & keep
        neg_cond = negative & keep
    conditional = pos_cond | neg_cond
    return {
        "positive": positive,
        "negative": negative,
        "conditional": conditional,
        "pos_cond": pos_cond,
        "neg_cond": neg_cond,
        "pos_uncond": positive & ~pos_cond,
        "neg_uncond": negative & ~neg_cond,
    }


def _masked_mean(values: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    maskf = mask.astype(jnp.float32)
    return jnp.sum(values * maskf) / (jnp.sum(maskf) + _EPS)


# --------------------------------------------------------------------------- #
# the offline-RL loss (called inside the jitted train step)
# --------------------------------------------------------------------------- #
def compute_offline_rl_loss(
    model: _model.BaseModel,
    rng: jax.Array,
    observation: _model.Observation,
    actions: _model.Actions,
    rl_info: dict[str, jnp.ndarray],
    cfg: _config.OfflineRLConfig,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """Return (scalar loss, metrics). ``cfg`` is static (closed over by the jit)."""
    route_rng, loss_rng, loss_rng2 = jax.random.split(rng, 3)

    advantage = rl_info["advantage"].astype(bool)
    adv_w = rl_info["advantage_weight"].astype(jnp.float32)
    is_demo = rl_info["is_demo"].astype(bool)
    b = advantage.shape[0]

    routing = cfg_routing_masks(
        advantage, cfg.cfg.positive_only_conditional, cfg.cfg.uncond_prob, route_rng
    )
    cond = routing["conditional"]
    uncond = ~cond

    # --- effective observation: CFG-routed prompt (or raw if CFG disabled) ----
    if cfg.cfg.enabled:
        raw_t, raw_m = observation.tokenized_prompt, observation.tokenized_prompt_mask
        pos_t, pos_m = rl_info["pos_tokens"], rl_info["pos_mask"]
        if cfg.cfg.positive_only_conditional:
            sel = routing["pos_cond"][:, None]
            eff_t = jnp.where(sel, pos_t, raw_t)
            eff_m = jnp.where(sel, pos_m, raw_m)
        else:
            neg_t, neg_m = rl_info["neg_tokens"], rl_info["neg_mask"]
            posb = routing["positive"][:, None]
            g_t = jnp.where(posb, pos_t, neg_t)
            g_m = jnp.where(posb, pos_m, neg_m)
            condb = cond[:, None]
            eff_t = jnp.where(condb, g_t, raw_t)
            eff_m = jnp.where(condb, g_m, raw_m)
        obs_eff = dataclasses.replace(
            observation, tokenized_prompt=eff_t, tokenized_prompt_mask=eff_m
        )
    else:
        obs_eff = observation

    per_sample = jnp.mean(model.compute_loss(loss_rng, obs_eff, actions, train=True), axis=-1)  # [b]

    if cfg.sft_aux.mode == "reuse_unconditional":
        w = awr_weight(adv_w, cfg.awr.beta, cfg.awr.w_max, cond) if cfg.awr.enabled else cond.astype(jnp.float32)
        rl_term = jnp.sum(w * per_sample) / b
        sft_term = jnp.sum(uncond.astype(jnp.float32) * per_sample) / b
        awr_mean = _masked_mean(w, cond)
    elif cfg.sft_aux.mode == "separate_forward":
        ones = jnp.ones((b,), jnp.float32)
        w = awr_weight(adv_w, cfg.awr.beta, cfg.awr.w_max, ones) if cfg.awr.enabled else ones
        rl_term = jnp.sum(w * per_sample) / b
        sft_term = jnp.array(0.0, jnp.float32)
        if cfg.sft_aux.weight > 0.0:
            per_sample_raw = jnp.mean(
                model.compute_loss(loss_rng2, observation, actions, train=True), axis=-1
            )
            anchor = is_demo if cfg.sft_aux.demo_only else jnp.ones_like(is_demo)
            sft_term = _masked_mean(per_sample_raw, anchor)
        awr_mean = jnp.mean(w)
    else:
        raise ValueError(f"sft_aux.mode must be reuse_unconditional|separate_forward, got {cfg.sft_aux.mode}")

    total = rl_term + cfg.sft_aux.weight * sft_term
    metrics = {
        "loss": total,
        "rl_loss": rl_term,
        "sft_loss": sft_term,
        "awr_weight_mean": awr_mean,
        "conditional_ratio": jnp.mean(cond.astype(jnp.float32)),
        "positive_ratio": jnp.mean(routing["positive"].astype(jnp.float32)),
    }
    return total, metrics


# --------------------------------------------------------------------------- #
# data: guidance tokenization + advantage attachment + loader
# --------------------------------------------------------------------------- #
@dataclasses.dataclass(frozen=True)
class TokenizePromptWithGuidance(_transforms.DataTransformFn):
    """Like TokenizePrompt, but also emits positive/negative advantage-conditioned
    prompt variants (``prompt + "\\nAdvantage: positive|negative"``)."""

    tokenizer: Any

    def __call__(self, data: dict) -> dict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")
        if not isinstance(prompt, str):
            prompt = prompt.item()
        tokens, mask = self.tokenizer.tokenize(prompt)
        pos_tokens, pos_mask = self.tokenizer.tokenize(f"{prompt}\nAdvantage: positive")
        neg_tokens, neg_mask = self.tokenizer.tokenize(f"{prompt}\nAdvantage: negative")
        return {
            **data,
            "tokenized_prompt": tokens,
            "tokenized_prompt_mask": mask,
            "tokenized_positive_guidance_prompt": pos_tokens,
            "tokenized_positive_guidance_prompt_mask": pos_mask,
            "tokenized_negative_guidance_prompt": neg_tokens,
            "tokenized_negative_guidance_prompt_mask": neg_mask,
        }


def _build_lerobot_dataset(data_config, model_config):
    """Build a local LeRobot dataset using the lerobot>=0.3 API (the openpi fork's
    ``create_dataset`` targets the older ``lerobot.common`` API and is incompatible
    with lerobot 0.3.x). Mirrors the validated load: repo_id is a label, ``root`` is
    the local dataset directory; fps/tasks are read from ``meta/``."""
    import json

    import lerobot.datasets.lerobot_dataset as lerobot_dataset

    root = Path(data_config.repo_id)
    info = json.loads((root / "meta" / "info.json").read_text())
    fps = float(info["fps"])
    delta_timestamps = {
        key: [t / fps for t in range(model_config.action_horizon)]
        for key in data_config.action_sequence_keys
    }
    ds = lerobot_dataset.LeRobotDataset(
        repo_id=f"pt4fm/{root.name}", root=str(root), delta_timestamps=delta_timestamps
    )
    if data_config.prompt_from_task:
        tasks = {}
        for line in (root / "meta" / "tasks.jsonl").read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                tasks[int(r["task_index"])] = r["task"]
        ds = _data_loader.TransformedDataset(ds, [_transforms.PromptFromLeRobotTask(tasks)])
    return ds


def _get_hf_dataset(dataset: Any) -> Any:
    cur = dataset
    while cur is not None:
        if hasattr(cur, "hf_dataset"):
            return cur.hf_dataset
        cur = getattr(cur, "_dataset", None)
    return None


def _load_advantages_lookup(repo_id: str, advantage_tag: str | None) -> dict[tuple[int, int], tuple]:
    import pandas as pd

    name = f"advantages_{advantage_tag}.parquet" if advantage_tag else "advantages.parquet"
    path = Path(repo_id) / "meta" / name
    if not path.exists():
        raise FileNotFoundError(
            f"Advantage file not found: {path}. Run PT4FM stages 1-3 (PyTorch) first, "
            f"or set offline_rl.advantage_tag correctly."
        )
    df = pd.read_parquet(path)
    ep = df["episode_index"].to_numpy().astype(int)
    fr = df["frame_index"].to_numpy().astype(int)
    adv = df["advantage"].to_numpy().astype(bool)
    wt = df["advantage_weight"].to_numpy().astype(float) if "advantage_weight" in df.columns else np.ones(len(df))
    demo = df["is_demo"].to_numpy().astype(bool) if "is_demo" in df.columns else np.ones(len(df), bool)
    return {(int(e), int(f)): (bool(a), float(w), bool(d)) for e, f, a, w, d in zip(ep, fr, adv, wt, demo)}


class _AdvantagePreservingDataset:
    """Adds advantage/advantage_weight/is_demo to each (post-transform) sample,
    looked up by (episode_index, frame_index) from the stage-3 parquet."""

    def __init__(self, base_dataset: Any, transformed_dataset: Any, lookup: dict[tuple[int, int], tuple]):
        self._t = transformed_dataset
        hf = _get_hf_dataset(base_dataset)
        if hf is None:
            raise ValueError("Cannot access underlying hf_dataset to align advantages.")
        ep = hf["episode_index"]
        fr = hf["frame_index"]
        self._by_index: dict[int, tuple] = {}
        missing = []
        for i in range(len(hf)):
            key = (int(ep[i]), int(fr[i]))
            if key in lookup:
                self._by_index[i] = lookup[key]
            else:
                missing.append(key)
        if missing:
            raise ValueError(
                f"{len(missing)} samples missing from advantages parquet (first 5: {missing[:5]}). "
                f"Re-run stage-3 on the same dataset."
            )

    def __len__(self) -> int:
        return len(self._t)

    def __getitem__(self, idx: int) -> dict:
        sample = self._t[idx]
        adv, wt, demo = self._by_index[idx]
        sample["advantage"] = np.asarray(adv)
        sample["advantage_weight"] = np.asarray(wt, dtype=np.float32)
        sample["is_demo"] = np.asarray(demo)
        return sample


class OfflineRLDataLoader:
    """Yields ``(Observation, actions, rl_info)``; ``rl_info`` carries the per-frame
    advantage fields (+ guidance tokens when CFG is enabled)."""

    def __init__(self, data_config, torch_loader, cfg_enabled: bool):
        self._data_config = data_config
        self._loader = torch_loader
        self._cfg_enabled = cfg_enabled

    def data_config(self):
        return self._data_config

    def __iter__(self):
        for batch in self._loader:
            obs = _model.Observation.from_dict(batch)
            rl_info = {
                "advantage": batch["advantage"],
                "advantage_weight": batch["advantage_weight"],
                "is_demo": batch["is_demo"],
            }
            if self._cfg_enabled:
                rl_info.update(
                    pos_tokens=batch["tokenized_positive_guidance_prompt"],
                    pos_mask=batch["tokenized_positive_guidance_prompt_mask"],
                    neg_tokens=batch["tokenized_negative_guidance_prompt"],
                    neg_mask=batch["tokenized_negative_guidance_prompt_mask"],
                )
            yield obs, batch["actions"], rl_info


def create_offline_rl_data_loader(config: _config.TrainConfig, *, sharding=None, shuffle: bool = True):
    """Build the offline-RL data loader (mirrors openpi.create_data_loader but attaches
    advantages and, when CFG is on, guidance-tokenized prompt variants)."""
    assert config.offline_rl is not None, "config.offline_rl must be set for offline-RL training"
    cfg = config.offline_rl
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.repo_id is None:
        raise ValueError("offline-RL requires a LeRobot repo_id (local path).")

    base_dataset = _build_lerobot_dataset(data_config, config.model)

    # Build the standard transform stack; swap TokenizePrompt -> guidance variant if CFG on.
    norm_stats = {} if data_config.norm_stats is None else data_config.norm_stats
    model_inputs = list(data_config.model_transforms.inputs)
    if cfg.cfg.enabled:
        swapped = []
        for t in model_inputs:
            if type(t).__name__ == "TokenizePrompt":
                swapped.append(TokenizePromptWithGuidance(tokenizer=t.tokenizer))
            else:
                swapped.append(t)
        model_inputs = swapped

    transformed = _data_loader.TransformedDataset(
        base_dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *model_inputs,
        ],
    )

    lookup = _load_advantages_lookup(data_config.repo_id, cfg.advantage_tag)
    final = _AdvantagePreservingDataset(base_dataset, transformed, lookup)

    torch_loader = _data_loader.TorchDataLoader(
        final,
        local_batch_size=config.batch_size // jax.process_count(),
        sharding=sharding,
        shuffle=shuffle,
        num_workers=config.num_workers,
        seed=config.seed,
    )
    return OfflineRLDataLoader(data_config, torch_loader, cfg_enabled=cfg.cfg.enabled)
